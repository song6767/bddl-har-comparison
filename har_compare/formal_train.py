"""Run a fixed-budget learning-rate sweep and the frozen multiseed training."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

from .runtime import metadata_fingerprint


MODELS = ("cnn", "rnn", "lstm")
LEARNING_RATES = (0.0003, 0.001, 0.003)
FORMAL_SEEDS = (42, 43, 44)
TRAINING_FILES = ("data.py", "models.py", "metrics.py", "runtime.py", "train.py",
                  "formal_train.py")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_once(path, content):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(content, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def protocol_for(data_dir, device):
    metadata = read_json(data_dir / "metadata.json")
    package = Path(__file__).resolve().parent
    return {
        "version": 1,
        "models": list(MODELS),
        "candidate_learning_rates": list(LEARNING_RATES),
        "tuning_seed": 41,
        "formal_seeds": list(FORMAL_SEEDS),
        "max_epochs": 50,
        "patience": 8,
        "batch_size": 64,
        "dropout": 0.2,
        "threads": 4,
        "optimizer": "Adam",
        "gradient_clip_norm": 1.0,
        "selection_metric": "val_macro_f1",
        "learning_rate_tie_break": "smallest learning rate",
        "checkpoint_tie_break": "earliest epoch",
        "device": device,
        "dataset_fingerprint": metadata_fingerprint(metadata),
        "sample_counts": metadata["sample_counts"],
        "split_subjects": metadata["split_subjects"],
        "test_plan": {
            "split": "official test",
            "evaluate": "all 9 frozen best-validation checkpoints",
            "metrics": ["accuracy", "macro_f1", "per_class_f1", "confusion_matrix"],
            "aggregate": "per-model arithmetic mean and sample SD over 3 seeds",
        },
        "training_source_sha256": {
            filename: sha256((package / filename).read_bytes()).hexdigest()
            for filename in TRAINING_FILES
        },
    }


def run_training(protocol, data_dir, folder, model, seed, rate, stage):
    expected = {
        "model": model, "seed": seed, "lr": rate, "stage": stage,
        "epochs": protocol["max_epochs"], "patience": protocol["patience"],
        "batch_size": protocol["batch_size"], "dropout": protocol["dropout"],
        "threads": protocol["threads"], "device": protocol["device"],
    }
    metrics_file = folder / "metrics.json"
    outputs = ("best.pt", "config.json", "history.csv", "learning_curves.png",
               "confusion_matrix.png")
    if not metrics_file.is_file() or any(not (folder / name).is_file() for name in outputs):
        if folder.exists() and any(folder.iterdir()):
            raise RuntimeError(f"Incomplete run at {folder}. Preserve it and move it aside before retrying.")
        command = [sys.executable, "-m", "har_compare.train",
                   "--data-dir", str(data_dir), "--output-dir", str(folder)]
        for key, value in expected.items():
            command.extend(("--" + key.replace("_", "-"), str(value)))
        print(f"Starting {stage}: {model}, seed {seed}, lr={rate:g}", flush=True)
        subprocess.run(command, check=True)
    else:
        print(f"Verifying existing {stage} run at {folder}", flush=True)
    record = read_json(metrics_file)
    if (record["status"] != "completed" or record["test_evaluated"] is not False
            or record["data_metadata_sha256"] != protocol["dataset_fingerprint"]
            or any(record["config"].get(key) != value for key, value in expected.items())):
        raise ValueError(f"Run does not match the protocol: {folder}")
    return record


def tuning_folder(root, model, rate):
    return root / "tuning" / model / f"lr_{rate:g}"


def read_selection(root, protocol):
    candidates = []
    chosen = {}
    for model in protocol["models"]:
        model_candidates = []
        for rate in protocol["candidate_learning_rates"]:
            path = tuning_folder(root, model, rate) / "metrics.json"
            row = read_json(path)
            if (row["status"] != "completed" or row["stage"] != "pilot"
                    or row["test_evaluated"] is not False
                    or row["model"] != model or row["seed"] != protocol["tuning_seed"]
                    or row["config"]["lr"] != rate
                    or row["data_metadata_sha256"] != protocol["dataset_fingerprint"]):
                raise ValueError(f"Tuning record does not match the protocol: {path}")
            model_candidates.append({
                "model": model,
                "learning_rate": rate,
                "val_macro_f1": row["val_macro_f1"],
                "val_accuracy": row["val_accuracy"],
                "best_epoch": row["best_epoch"],
                "epochs_completed": row["epochs_completed"],
                "metrics_sha256": sha256(path.read_bytes()).hexdigest(),
            })
        best = min(model_candidates, key=lambda row: (-row["val_macro_f1"], row["learning_rate"]))
        chosen[model] = best["learning_rate"]
        candidates.extend(model_candidates)
    return {
        "protocol_sha256": sha256((root / "protocol.json").read_bytes()).hexdigest(),
        "learning_rates": chosen,
        "candidates": candidates,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("tune", "train"))
    parser.add_argument("--root", type=Path, default=Path("runs/formal_v1"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    protocol_path, selection_path = root / "protocol.json", root / "selection.json"
    if (root / "freeze.json").exists() or (root / "test").exists():
        raise RuntimeError("The experiment is frozen for testing. Training requires a new root.")
    protocol = protocol_for(args.data_dir, args.device)
    if args.action == "tune" and not protocol_path.exists():
        write_once(protocol_path, protocol)
    if read_json(protocol_path) != protocol:
        raise ValueError("Protocol, training source or data changed. Use the original settings or a new root.")
    if args.action == "tune":
        for model in MODELS:
            for rate in LEARNING_RATES:
                run_training(protocol, args.data_dir, tuning_folder(root, model, rate),
                             model, protocol["tuning_seed"], rate, "pilot")
        selection = read_selection(root, protocol)
        if selection_path.exists():
            if read_json(selection_path) != selection:
                raise ValueError("Existing learning-rate selection differs from tuning records.")
        else:
            write_once(selection_path, selection)
        print(f"Validation-selected learning rates: {selection['learning_rates']}", flush=True)
    else:
        selection = read_json(selection_path)
        if selection != read_selection(root, protocol):
            raise ValueError("Learning-rate selection does not match tuning records.")
        for model in MODELS:
            for seed in FORMAL_SEEDS:
                run_training(protocol, args.data_dir, root / "training" / model / f"seed_{seed}",
                             model, seed, selection["learning_rates"][model], "formal")
        print("Completed 9 formal train/validation runs. Ready to freeze checkpoints.", flush=True)


if __name__ == "__main__":
    main()
