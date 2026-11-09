"""Freeze completed formal runs and evaluate them as one official-test batch."""

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from .runtime import (DATA_FINGERPRINT_SCHEMA_VERSION, data_loader, environment_info,
                      evaluate_model, metadata_fingerprint, resolve_device, seed_everything,
                      write_json)

import torch

from .data import load_data
from .models import build_model


def read_json(path):
    payload = path.read_bytes()
    return json.loads(payload), hashlib.sha256(payload).hexdigest()


def collect_runs(root, data_dir):
    """Verify the complete train/validation record without opening test.npz."""
    protocol, protocol_hash = read_json(root / "protocol.json")
    selection, selection_hash = read_json(root / "selection.json")
    metadata, _ = read_json(data_dir / "metadata.json")
    fingerprint = metadata_fingerprint(metadata)
    models = protocol["models"]
    seeds = protocol["formal_seeds"]
    rates = selection["learning_rates"]
    if sorted(models) != ["cnn", "lstm", "rnn"] or sorted(seeds) != [42, 43, 44]:
        raise ValueError("The formal comparison must contain CNN, RNN and LSTM with seeds 42, 43 and 44.")
    if set(rates) != set(models) or any(not math.isfinite(lr) or lr <= 0 for lr in rates.values()):
        raise ValueError("Selection must specify one positive learning rate for each model.")
    if fingerprint != protocol["dataset_fingerprint"]:
        raise ValueError("Prepared data do not match the experiment protocol.")
    provenance = {
        "schema_version": 1,
        "dataset_fingerprint": fingerprint,
        "official_test_sha256": metadata["processed_sha256"]["test"],
        "protocol_sha256": protocol_hash,
        "selection_sha256": selection_hash,
        "checkpoints": [],
    }
    checkpoints = []
    for model in models:
        for seed in seeds:
            run = Path("training") / model / f"seed_{seed}"
            metrics, metrics_hash = read_json(root / run / "metrics.json")
            weights = (root / run / "best.pt").read_bytes()
            checkpoint = torch.load(io.BytesIO(weights), map_location="cpu", weights_only=True)
            expected = {
                "model": model, "seed": seed, "stage": "formal", "lr": rates[model],
                "epochs": protocol["max_epochs"], "patience": protocol["patience"],
                "batch_size": protocol["batch_size"], "dropout": protocol["dropout"],
                "device": protocol["device"], "threads": protocol.get("threads", 4),
                "optimizer": "Adam", "gradient_clip_norm": 1.0,
                "selection_metric": "val_macro_f1", "test_evaluated": False,
            }
            for field, value in expected.items():
                if checkpoint["config"].get(field) != value or metrics["config"].get(field) != value:
                    raise ValueError(f"{run}: {field} does not match the formal configuration.")
            for artifact in (checkpoint, metrics):
                if (artifact["data_metadata_sha256"] != fingerprint or
                        artifact["data_fingerprint_schema_version"] != DATA_FINGERPRINT_SCHEMA_VERSION):
                    raise ValueError(f"{run}: data fingerprint mismatch.")
            if (metrics["status"] != "completed" or metrics["test_evaluated"] is not False or
                    metrics["stage"] != "formal" or metrics["model"] != model or metrics["seed"] != seed or
                    checkpoint["model_name"] != model or checkpoint["epoch"] != metrics["best_epoch"] or
                    not 1 <= checkpoint["epoch"] <= metrics["epochs_completed"] <= protocol["max_epochs"] or
                    checkpoint["best_val_macro_f1"] != metrics["val_macro_f1"] or
                    checkpoint["started_at_utc"] != metrics["started_at_utc"]):
                raise ValueError(f"{run}: checkpoint and completed validation record do not agree.")
            provenance["checkpoints"].append({
                "model": model, "seed": seed, "learning_rate": rates[model],
                "epoch": checkpoint["epoch"], "path": (run / "best.pt").as_posix(),
                "sha256": hashlib.sha256(weights).hexdigest(),
                "metrics_path": (run / "metrics.json").as_posix(), "metrics_sha256": metrics_hash,
            })
            checkpoints.append(checkpoint)
    return protocol, provenance, checkpoints


def freeze(root, data_dir):
    destination = root / "freeze.json"
    if destination.exists() or (root / "test").exists():
        raise FileExistsError("This experiment already has a freeze or official-test results.")
    _, provenance, _ = collect_runs(root, data_dir)
    provenance["frozen_at_utc"] = datetime.now(timezone.utc).isoformat()
    with TemporaryDirectory(prefix=".freeze-", dir=root) as temporary:
        staged = Path(temporary) / destination.name
        write_json(staged, provenance)
        # Publish the complete file without replacing another freeze.
        os.link(staged, destination)
    print(f"Froze {len(provenance['checkpoints'])} checkpoints in {destination}; official test unopened.", flush=True)


def result_path(record):
    return Path(record["model"]) / f"seed_{record['seed']}.json"


def evaluate(root, data_dir, requested_device):
    manifest, freeze_hash = read_json(root / "freeze.json")
    protocol, current, checkpoints = collect_runs(root, data_dir)
    if {key: value for key, value in manifest.items() if key != "frozen_at_utc"} != current:
        raise ValueError("Freeze verification failed. Protocol, selection, data or training artifacts changed.")
    device = resolve_device(requested_device or protocol["device"])
    if str(device) != protocol["device"]:
        raise ValueError("Use the device type recorded in the formal protocol.")
    destination = root / "test"
    identities = []
    for record in manifest["checkpoints"]:
        identities.append({
            "model": record["model"], "seed": record["seed"], "split": "test", "stage": "formal",
            "checkpoint_epoch": record["epoch"], "checkpoint_sha256": record["sha256"],
            "freeze_sha256": freeze_hash, "data_metadata_sha256": manifest["dataset_fingerprint"],
            "data_fingerprint_schema_version": DATA_FINGERPRINT_SCHEMA_VERSION, "test_evaluated": True,
        })
    if destination.exists():
        for identity in identities:
            saved, _ = read_json(destination / result_path(identity))
            if any(saved.get(key) != value for key, value in identity.items()) or not saved.get("metrics"):
                raise ValueError("Existing official-test results do not match the freeze.")
        print(f"Official test batch already published in {destination}.", flush=True)
        return

    # Build all frozen models before the first test-data access.
    seed_everything(protocol["formal_seeds"][0], protocol.get("threads", 4))
    models = []
    for checkpoint in checkpoints:
        model = build_model(checkpoint["model_name"], dropout=protocol["dropout"])
        model.load_state_dict(checkpoint["model_state_dict"])
        models.append(model)
    dataset = load_data(data_dir, include_test=True)
    if (metadata_fingerprint(dataset["metadata"]) != manifest["dataset_fingerprint"] or
            dataset["metadata"]["processed_sha256"]["test"] != manifest["official_test_sha256"]):
        raise ValueError("Loaded official test does not match the freeze.")
    loader = data_loader(dataset["test"], protocol["batch_size"], device)
    environment = environment_info(device)
    with TemporaryDirectory(prefix=".formal-test-", dir=root) as temporary:
        staged = Path(temporary) / "test"
        for identity, model in zip(identities, models):
            report = {**identity, "environment": environment,
                      "metrics": evaluate_model(model.to(device), loader, device)}
            write_json(staged / result_path(identity), report)
        staged.rename(destination)
    print(f"Published {len(identities)} frozen evaluations in {destination}.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "evaluate"])
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--data-dir", default=Path("data/processed"), type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], help="Device for evaluation; defaults to the protocol.")
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.root, args.data_dir)
    else:
        evaluate(args.root, args.data_dir, args.device)


if __name__ == "__main__":
    main()
