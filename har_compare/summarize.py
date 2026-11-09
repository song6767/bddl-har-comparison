"""Export lightweight validation results for review and version control."""

import argparse
import csv
import json
from pathlib import Path
import shutil

from .runtime import write_json


def summarize(runs_root="runs/smoke", output_dir="reports/smoke"):
    root, output = Path(runs_root), Path(output_dir)
    paths = sorted(root.rglob("metrics.json"))
    if not paths:
        raise FileNotFoundError(f"No completed metrics.json files in {root}")
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if any(row.get("status") != "completed" or row.get("test_evaluated") is not False for row in records):
        raise ValueError("This exporter accepts completed train/validation runs only.")
    if len({row["data_metadata_sha256"] for row in records}) != 1:
        raise ValueError("Refusing to compare runs using different data/split/normalization.")
    if len({row["stage"] for row in records}) != 1:
        raise ValueError("Keep smoke, pilot and formal results in separate reports.")
    keys = [(row["model"], row["seed"]) for row in records]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate model/seed runs: export different configurations separately.")
    output.mkdir(parents=True, exist_ok=True)
    fields = ["model", "seed", "best_epoch", "epochs_completed", "parameters", "val_accuracy",
              "val_macro_f1", "fit_seconds", "peak_torch_allocated_mib", "inference_batch_ms"]
    rows = []
    for source, record in zip(paths, records):
        row = {key: record[key] for key in fields if key != "inference_batch_ms"}
        row["inference_batch_ms"] = record["inference"]["batch_ms"]
        rows.append(row)
        destination = output / record["model"] / f"seed_{record['seed']}"
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("metrics.json", "history.csv", "learning_curves.png", "confusion_matrix.png"):
            shutil.copy2(source.parent / name, destination / name)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "summary.json", {"stage": records[0]["stage"], "test_evaluated": False,
               "data_metadata_sha256": records[0]["data_metadata_sha256"], "runs": rows})
    config = json.loads((paths[0].parent / "config.json").read_text(encoding="utf-8"))
    write_json(output / "data_manifest.json", config["data_metadata"])
    print(f"Exported {len(records)} validation runs to {output}; datasets and weights excluded.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", default="runs/smoke")
    parser.add_argument("--output-dir", default="reports/smoke")
    args = parser.parse_args()
    summarize(args.runs_root, args.output_dir)


if __name__ == "__main__":
    main()
