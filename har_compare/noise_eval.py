"""Evaluate a predeclared noise experiment using the nine frozen checkpoints."""

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

from .formal_test import collect_runs, read_json, result_path
from .runtime import (data_loader, environment_info, evaluate_model,
                      metadata_fingerprint, resolve_device, seed_everything, write_json)

import numpy as np

from .data import load_data
from .models import build_model


SNR_DB = (20, 10)
NOISE_SEED = 20261109


def protocol_for(root, data_dir):
    freeze, freeze_hash = read_json(root / "freeze.json")
    metadata, _ = read_json(data_dir / "metadata.json")
    if (metadata_fingerprint(metadata) != freeze["dataset_fingerprint"]
            or metadata["processed_sha256"]["test"] != freeze["official_test_sha256"]):
        raise ValueError("Prepared test data do not match the freeze.")
    return {
        "version": 1,
        "snr_db": list(SNR_DB),
        "noise_seed": NOISE_SEED,
        "generator": "NumPy Generator(PCG64)",
        "input_domain": "Standardized official UCI HAR inertial windows",
        "noise": "Independent standard-normal draws before RMS scaling per window and channel",
        "power": "Mean square over the 128 time points, including DC",
        "scaling": "Unit-RMS template times clean-window channel RMS times 10**(-snr_db/20)",
        "sharing": "Identical noisy input across all nine checkpoints per level; same template across levels",
        "zero_power": "Unperturbed",
        "postprocess": "Cast to float32; no clipping or restandardization",
        "baseline": "Recompute clean scores and confirm against the saved official-test metrics",
        "comparison": "100 * (clean - noisy) paired within checkpoint, then mean and sample SD over 3 seeds",
        "freeze_sha256": freeze_hash,
        "dataset_fingerprint": freeze["dataset_fingerprint"],
        "official_test_sha256": freeze["official_test_sha256"],
        "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def rms(x):
    return np.sqrt(np.mean(x * x, axis=1, keepdims=True))


def noise_template(shape):
    rng = np.random.Generator(np.random.PCG64(NOISE_SEED))
    draws = rng.standard_normal(shape)
    return draws / rms(draws)


def perturb(clean, template, snr_db):
    signal = clean.astype(np.float64)
    signal_rms = rms(signal)
    noisy = (signal + template * signal_rms * 10 ** (-snr_db / 20)).astype(np.float32)
    # Measure the perturbation after rounding to the actual float32 model input.
    noise_rms = rms(noisy.astype(np.float64) - signal)
    nonzero = signal_rms > 0
    actual_snr = 20 * np.log10(signal_rms[nonzero] / noise_rms[nonzero])
    if not np.allclose(actual_snr, snr_db, rtol=0, atol=0.001):
        raise ValueError(f"Realized SNR does not match {snr_db} dB.")
    info = {
        "snr_db": snr_db,
        "noisy_input_sha256": hashlib.sha256(noisy.astype("<f4", copy=False).tobytes(order="C")).hexdigest(),
        "realized_snr_min_db": float(actual_snr.min()),
        "realized_snr_median_db": float(np.median(actual_snr)),
        "realized_snr_max_db": float(actual_snr.max()),
        "nonzero_power_window_channels": int(nonzero.sum()),
        "zero_power_window_channels": int(nonzero.size - nonzero.sum()),
    }
    return noisy, info


def plan(root, data_dir, directory):
    protocol = protocol_for(root, data_dir)
    protocol["planned_at_utc"] = datetime.now(timezone.utc).isoformat()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "protocol.json"
    if path.exists() or (directory / "results.json").exists():
        raise FileExistsError(f"Noise experiment already planned in {directory}.")
    with TemporaryDirectory(prefix=".plan-", dir=directory) as temporary:
        staged = Path(temporary) / path.name
        write_json(staged, protocol)
        # A hard link publishes exclusively, preserving the pre-evaluation plan.
        path.hardlink_to(staged)
    print(f"Locked noise protocol at {path}.", flush=True)


def evaluate(root, data_dir, directory, requested_device):
    protocol, protocol_hash = read_json(directory / "protocol.json")
    if {key: value for key, value in protocol.items() if key != "planned_at_utc"} != protocol_for(root, data_dir):
        raise ValueError("Noise plan, freeze, data or evaluator code changed.")
    destination = directory / "results.json"
    if destination.exists():
        raise FileExistsError(f"Noise scores already exist at {destination}.")
    manifest, freeze_hash = read_json(root / "freeze.json")
    formal_protocol, current, checkpoints = collect_runs(root, data_dir)
    if {key: value for key, value in manifest.items() if key != "frozen_at_utc"} != current:
        raise ValueError("The frozen checkpoints or their provenance changed.")
    device = resolve_device(requested_device or formal_protocol["device"])
    if str(device) != formal_protocol["device"]:
        raise ValueError("Use the same device type as the formal experiment.")

    seed_everything(formal_protocol["formal_seeds"][0], formal_protocol["threads"])
    models, baselines = [], []
    for record, checkpoint in zip(manifest["checkpoints"], checkpoints):
        saved, clean_hash = read_json(root / "test" / result_path(record))
        expected = {
            "model": record["model"], "seed": record["seed"], "split": "test", "stage": "formal",
            "checkpoint_epoch": record["epoch"], "checkpoint_sha256": record["sha256"],
            "freeze_sha256": freeze_hash, "data_metadata_sha256": manifest["dataset_fingerprint"],
            "test_evaluated": True,
        }
        if any(saved.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Clean test record does not match the freeze: {result_path(record)}")
        model = build_model(record["model"], dropout=formal_protocol["dropout"])
        model.load_state_dict(checkpoint["model_state_dict"])
        models.append(model.to(device))
        baselines.append((saved["metrics"], clean_hash))

    dataset = load_data(data_dir, include_test=True)
    metadata, clean = dataset["metadata"], dataset["test"]
    if (metadata_fingerprint(metadata) != manifest["dataset_fingerprint"]
            or metadata["processed_sha256"]["test"] != manifest["official_test_sha256"]
            or len(clean["y"]) != formal_protocol["sample_counts"]["test"]):
        raise ValueError("Loaded test data differ from the frozen experiment.")
    batch_size = formal_protocol["batch_size"]
    clean_loader = data_loader(clean, batch_size, device)
    runs = []
    for record, model, (baseline, clean_hash) in zip(manifest["checkpoints"], models, baselines):
        metrics = evaluate_model(model, clean_loader, device)
        for key in ("n_samples", "confusion_matrix", "accuracy", "macro_f1"):
            if metrics[key] != baseline[key]:
                raise ValueError(f"Clean baseline mismatch for {result_path(record)}: {key}")
        runs.append({
            "model": record["model"], "seed": record["seed"], "checkpoint_epoch": record["epoch"],
            "checkpoint_sha256": record["sha256"], "clean_result_sha256": clean_hash,
            "clean": metrics, "noisy": [],
        })
    print("Reproduced the clean confusion matrix and scores for all nine checkpoints.", flush=True)

    template = noise_template(clean["x"].shape)
    perturbations = []
    for snr_db in SNR_DB:
        noisy, info = perturb(clean["x"], template, snr_db)
        perturbations.append(info)
        loader = data_loader({**clean, "x": noisy}, batch_size, device)
        for model, run in zip(models, runs):
            metrics = evaluate_model(model, loader, device)
            run["noisy"].append({"snr_db": snr_db, "metrics": metrics})
        print(f"Scored all nine checkpoints on the same {snr_db} dB tensor.", flush=True)
    results = {
        "protocol": protocol, "protocol_sha256": protocol_hash,
        "freeze_sha256": freeze_hash, "dataset_fingerprint": manifest["dataset_fingerprint"],
        "official_test_sha256": manifest["official_test_sha256"],
        "test_subjects": len(metadata["split_subjects"]["test"]), "test_windows": len(clean["y"]),
        "environment": environment_info(device), "perturbations": perturbations,
        "runs": runs,
    }
    with TemporaryDirectory(prefix=".results-", dir=directory) as temporary:
        staged = Path(temporary) / destination.name
        write_json(staged, results)
        destination.hardlink_to(staged)
    print(f"Saved the paired noise evaluations at {destination}.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "evaluate"))
    parser.add_argument("--root", type=Path, default=Path("runs/formal_v1"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    directory = args.root / "noise_v1"
    if args.command == "plan":
        plan(args.root, args.data_dir, directory)
    else:
        evaluate(args.root, args.data_dir, directory, args.device)


if __name__ == "__main__":
    main()
