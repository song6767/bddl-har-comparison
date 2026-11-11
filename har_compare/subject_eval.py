"""Describe per-subject errors of all nine already frozen official-test models."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import load_data
from .formal_test import collect_runs, read_json, result_path
from .metrics import classification_metrics
from .models import build_model
from .runtime import (data_loader, environment_info, metadata_fingerprint,
                      resolve_device, seed_everything, write_json)


MODELS = ("cnn", "rnn", "lstm")
SEEDS = (42, 43, 44)
OUTPUT_FILES = ("subject_results.json", "subject_scores.csv", "subject_scores.png",
                "subject_errors.png")


def grouped_metrics(truth, prediction, subjects):
    """Use the same fixed six classes for each person and verify the partition."""
    truth, prediction, subjects = map(np.asarray, (truth, prediction, subjects))
    pooled = classification_metrics(truth, prediction, num_classes=6)
    if subjects.shape != truth.shape or not np.issubdtype(subjects.dtype, np.integer):
        raise ValueError("Subjects must be an integer vector aligned with the labels")
    groups = []
    for subject in np.unique(subjects):
        selected = subjects == subject
        groups.append({"subject": int(subject),
                       "metrics": classification_metrics(truth[selected], prediction[selected], 6)})
    combined = np.sum([item["metrics"]["confusion_matrix"] for item in groups], axis=0)
    if not np.array_equal(combined, pooled["confusion_matrix"]):
        raise ValueError("Per-subject confusion matrices do not reconstruct the pooled result")
    return pooled, groups


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "sample_sd": float(values.std(ddof=1))}


def summarize(runs):
    """Keep person-level summaries separate from pooling all windows together."""
    if sorted((r["model"], r["seed"]) for r in runs) != sorted(
            (m, s) for m in MODELS for s in SEEDS):
        raise ValueError("Expected exactly one run for each of three models and three seeds")
    subjects = [item["subject"] for item in runs[0]["subjects"]]
    support = {item["subject"]: item["metrics"]["support"] for item in runs[0]["subjects"]}
    for run in runs:
        if [item["subject"] for item in run["subjects"]] != subjects:
            raise ValueError("Every checkpoint must evaluate the same ordered subject set")
        if any(item["metrics"]["support"] != support[item["subject"]] for item in run["subjects"]):
            raise ValueError("Subject class support differs between checkpoints")
    models, rows = {}, []
    for model in MODELS:
        matching = sorted((r for r in runs if r["model"] == model), key=lambda r: r["seed"])
        subject_rows = []
        for index, subject in enumerate(subjects):
            scores = [r["subjects"][index]["metrics"] for r in matching]
            accuracy, f1 = describe([x["accuracy"] for x in scores]), describe([x["macro_f1"] for x in scores])
            matrix = np.sum([x["confusion_matrix"] for x in scores], axis=0)
            errors = int(matrix.sum() - matrix.trace())
            sit_stand = int(matrix[3, 4] + matrix[4, 3])
            item = {"subject": subject, "windows": scores[0]["n_samples"],
                    "accuracy": accuracy, "macro_f1": f1,
                    "confusion_counts_sum_across_seeds": matrix.tolist(),
                    "total_errors_sum_across_seeds": errors,
                    "sit_stand_errors_sum_across_seeds": sit_stand,
                    "sit_stand_share_of_errors_pct": 100 * sit_stand / errors if errors else None,
                    "sit_stand_error_rate_pct": 100 * sit_stand / int(matrix.sum()),
                    "other_error_rate_pct": 100 * (errors - sit_stand) / int(matrix.sum())}
            subject_rows.append(item)
            rows.append({"model": model, "subject": subject, "windows": item["windows"],
                         "accuracy_mean_pct": 100 * accuracy["mean"],
                         "accuracy_sample_sd_pct": 100 * accuracy["sample_sd"],
                         "macro_f1_mean_pct": 100 * f1["mean"],
                         "macro_f1_sample_sd_pct": 100 * f1["sample_sd"],
                         **{key: item[key] for key in (
                             "total_errors_sum_across_seeds", "sit_stand_errors_sum_across_seeds",
                             "sit_stand_share_of_errors_pct", "sit_stand_error_rate_pct", "other_error_rate_pct")}})
        models[model] = {
            "pooled_window_accuracy": describe([r["pooled"]["accuracy"] for r in matching]),
            "pooled_window_macro_f1": describe([r["pooled"]["macro_f1"] for r in matching]),
            "equal_subject_accuracy": describe([
                np.mean([x["metrics"]["accuracy"] for x in r["subjects"]]) for r in matching]),
            "equal_subject_macro_f1": describe([
                np.mean([x["metrics"]["macro_f1"] for x in r["subjects"]]) for r in matching]),
            "subjects": subject_rows,
        }
    return models, rows


def plots(output, models):
    subjects = models[MODELS[0]]["subjects"]
    means = np.array([[100 * item["macro_f1"]["mean"] for item in models[m]["subjects"]] for m in MODELS])
    sds = np.array([[100 * item["macro_f1"]["sample_sd"] for item in models[m]["subjects"]] for m in MODELS])
    fig, axis = plt.subplots(figsize=(11.4, 3.6), layout="constrained")
    im = axis.imshow(means, cmap="YlGnBu", vmin=5 * np.floor(means.min() / 5), vmax=100, aspect="auto")
    axis.set(xticks=range(len(subjects)), yticks=range(len(MODELS)),
             xticklabels=[f"{x['subject']}\n(n={x['windows']})" for x in subjects],
             yticklabels=[m.upper() for m in MODELS], xlabel="Official-test subject (unique windows)")
    for row in range(len(MODELS)):
        for col in range(len(subjects)):
            axis.text(col, row, f"{means[row,col]:.1f}\n±{sds[row,col]:.1f}", ha="center", va="center",
                      color="white" if im.norm(means[row,col]) > .58 else "black", fontsize=10)
    fig.colorbar(im, ax=axis, label="Macro-F1 (%)", shrink=.88)
    fig.suptitle("Per-subject Macro-F1 · mean ± sample SD over 3 frozen training seeds")
    fig.savefig(output / "subject_scores.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.0), sharey=True, layout="constrained")
    for axis, model in zip(axes, MODELS):
        items = models[model]["subjects"]
        sit = [x["sit_stand_error_rate_pct"] for x in items]
        other = [x["other_error_rate_pct"] for x in items]
        x = np.arange(len(items))
        axis.bar(x, sit, color="#D55E00", label="Sit ↔ stand")
        axis.bar(x, other, bottom=sit, color="#0072B2", label="Other errors")
        axis.set(xticks=x, xticklabels=[i["subject"] for i in items], title=model.upper(),
                 xlabel="Official-test subject")
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Errors / all predictions for that subject (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncols=2, loc="outside lower center", frameon=False)
    fig.suptitle("Per-subject error composition · counts pooled across 3 frozen checkpoints")
    fig.savefig(output / "subject_errors.png", dpi=180)
    plt.close(fig)


def evaluate(root, data_dir, output, requested_device):
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in OUTPUT_FILES):
        raise FileExistsError("Subject analysis already exists; choose a new --output directory")
    manifest, freeze_hash = read_json(root / "freeze.json")
    protocol, current, checkpoints = collect_runs(root, data_dir)
    if {k: v for k, v in manifest.items() if k != "frozen_at_utc"} != current:
        raise ValueError("Frozen checkpoints, configuration or provenance changed")
    for name, expected in protocol["training_source_sha256"].items():
        if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen training source changed: {name}")
    device = resolve_device(requested_device or protocol["device"])
    if str(device) != protocol["device"]:
        raise ValueError("Use the device type recorded in the formal experiment")
    seed_everything(SEEDS[0], protocol["threads"])
    data = load_data(data_dir, include_test=True)
    if (metadata_fingerprint(data["metadata"]) != manifest["dataset_fingerprint"] or
            data["metadata"]["processed_sha256"]["test"] != manifest["official_test_sha256"]):
        raise ValueError("Test data no longer match the frozen comparison")
    test = data["test"]
    loader = data_loader(test, protocol["batch_size"], device, shuffle=False)
    runs = []
    for record, checkpoint in zip(manifest["checkpoints"], checkpoints):
        baseline, baseline_hash = read_json(root / "test" / result_path(record))
        expected = {"model": record["model"], "seed": record["seed"],
                    "checkpoint_sha256": record["sha256"], "freeze_sha256": freeze_hash,
                    "data_metadata_sha256": manifest["dataset_fingerprint"],
                    "split": "test", "test_evaluated": True}
        if any(baseline.get(k) != v for k, v in expected.items()):
            raise ValueError("Official-test baseline does not belong to this frozen model")
        model = build_model(record["model"], dropout=protocol["dropout"]).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        predictions, truth = [], []
        with torch.inference_mode():
            for x, y in loader:
                predictions.append(model(x.to(device, non_blocking=True)).argmax(1).cpu().numpy())
                truth.append(y.numpy())
        truth, predictions = np.concatenate(truth), np.concatenate(predictions)
        if not np.array_equal(truth, test["y"]):
            raise ValueError("Prediction order differs from subject metadata")
        pooled, groups = grouped_metrics(truth, predictions, test["subjects"])
        for key in ("n_samples", "confusion_matrix", "accuracy", "macro_f1"):
            if pooled[key] != baseline["metrics"][key]:
                raise ValueError(f"Recomputed clean baseline differs: {record['model']} {record['seed']} {key}")
        runs.append({"model": record["model"], "seed": record["seed"],
                     "checkpoint_sha256": record["sha256"], "baseline_sha256": baseline_hash,
                     "prediction_sha256": hashlib.sha256(predictions.astype("<i8").tobytes()).hexdigest(),
                     "pooled": pooled, "subjects": groups})
        print(f"Verified {record['model']} seed {record['seed']}: {len(groups)} subjects", flush=True)
    models, rows = summarize(runs)
    result = {"schema_version": 1, "analysis": "Post-test descriptive subject analysis of frozen models",
              "scope": "No retraining, retuning or additional independent test set; all nine original test subjects retained",
              "metric_scope": "Fixed six-class Macro-F1 for every subject; equal-subject means are not pooled-window Macro-F1",
              "uncertainty_scope": "Sample SD across three training seeds on the same subject or same nine-subject mean; not a population confidence interval",
              "count_scope": "Summed confusion counts repeat each window for three checkpoints; unique subjects remain nine",
              "freeze_sha256": freeze_hash, "dataset_fingerprint": manifest["dataset_fingerprint"],
              "official_test_sha256": manifest["official_test_sha256"],
              "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "baseline_reproduced_for_all_checkpoints": True,
              "environment": environment_info(device), "models": models, "runs": runs}
    with TemporaryDirectory(prefix=".subject-", dir=output) as temp:
        staged = Path(temp)
        write_json(staged / "subject_results.json", result)
        with (staged / "subject_scores.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        plots(staged, models)
        for name in OUTPUT_FILES:
            # Publish without silently replacing previous analysis files.
            (output / name).hardlink_to(staged / name)
    print(f"Published 81 subject/checkpoint groups and 27 subject/model summaries to {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/formal_v1"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--output", type=Path, default=Path("reports/formal"))
    args = parser.parse_args()
    evaluate(args.root, args.data_dir, args.output, args.device)


if __name__ == "__main__":
    main()
