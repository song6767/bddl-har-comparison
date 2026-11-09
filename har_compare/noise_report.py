"""Summarize paired clean and noisy scores for the frozen UCI HAR models."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
from statistics import mean, stdev
from tempfile import TemporaryDirectory

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODELS = ("cnn", "rnn", "lstm")
SEEDS = (42, 43, 44)
METRICS = ("accuracy", "macro_f1")
LABELS = ("Accuracy", "Macro-F1")
COLORS = ("#0072B2", "#D55E00", "#009E73")
MARKERS = ("o", "s", "^")


def describe(values):
    return {"mean": mean(values), "sample_sd": stdev(values)}


def read_comparison(path):
    protocol_path = path.with_name("protocol.json")
    protocol_bytes = protocol_path.read_bytes()
    data = json.loads(path.read_bytes())
    protocol = data["protocol"]
    if json.loads(protocol_bytes) != protocol or hashlib.sha256(protocol_bytes).hexdigest() != data["protocol_sha256"]:
        raise ValueError("Noise results do not match the saved protocol")
    levels = protocol["snr_db"]
    if levels != [20, 10] or sorted(row["snr_db"] for row in data["perturbations"]) != sorted(levels):
        raise ValueError("Expected one shared input at each planned SNR")
    runs = {(row["model"], row["seed"]): row for row in data["runs"]}
    expected = {(model, seed) for model in MODELS for seed in SEEDS}
    if set(runs) != expected or len(data["runs"]) != len(expected):
        raise ValueError("Expected one result for each of the nine frozen runs")
    return data, runs, levels


def summarize(data, runs, levels):
    rows, summaries = [], {}
    for model in MODELS:
        matrices = {condition: np.zeros((6, 6), dtype=np.int64) for condition in ("clean", *levels)}
        model_rows = []
        for seed in SEEDS:
            run = runs[model, seed]
            conditions = {item["snr_db"]: item["metrics"] for item in run["noisy"]}
            if set(conditions) != set(levels) or len(run["noisy"]) != len(levels):
                raise ValueError(f"Incomplete noisy evaluation for {model}, seed {seed}")
            conditions["clean"] = run["clean"]
            for condition, scores in conditions.items():
                counts = np.asarray(scores["confusion_matrix"], dtype=np.int64)
                if counts.shape != (6, 6) or np.any(counts < 0) or counts.sum() != data["test_windows"]:
                    raise ValueError(f"Invalid confusion matrix for {model}, seed {seed}, condition {condition}")
                matrices[condition] += counts
            for snr in levels:
                row = {"model": model, "seed": seed, "snr_db": snr,
                       "checkpoint_epoch": run["checkpoint_epoch"], "checkpoint_sha256": run["checkpoint_sha256"]}
                for metric in METRICS:
                    clean = run["clean"][metric]
                    noisy = conditions[snr][metric]
                    if not (0 <= clean <= 1 and 0 <= noisy <= 1):
                        raise ValueError("Accuracy and Macro-F1 must be scores in [0, 1]")
                    row[f"clean_{metric}_pct"] = 100 * clean
                    row[f"noisy_{metric}_pct"] = 100 * noisy
                    row[f"{metric}_drop_pp"] = 100 * (clean - noisy)
                model_rows.append(row)
        rows.extend(model_rows)
        summaries[model] = {
            "clean": {metric: describe([row[f"clean_{metric}_pct"] for row in model_rows if row["snr_db"] == levels[0]])
                      for metric in METRICS},
            "noisy": {
                str(snr): {metric: {
                    "score_pct": describe([row[f"noisy_{metric}_pct"] for row in model_rows if row["snr_db"] == snr]),
                    "paired_drop_pp": describe([row[f"{metric}_drop_pp"] for row in model_rows if row["snr_db"] == snr]),
                } for metric in METRICS}
                for snr in levels
            },
            "confusion_counts_summed_across_seeds": {str(key): value.tolist() for key, value in matrices.items()},
        }
    return rows, summaries


def plot_scores(output, summary, levels):
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2), sharey=True, layout="constrained")
    x = np.arange(1 + len(levels))
    for axis, metric, label in zip(axes, METRICS, LABELS):
        for model, color, marker, offset in zip(MODELS, COLORS, MARKERS, (-0.06, 0, 0.06)):
            stats = [summary[model]["clean"][metric]] + [
                summary[model]["noisy"][str(level)][metric]["score_pct"] for level in levels
            ]
            axis.errorbar(x + offset, [item["mean"] for item in stats], yerr=[item["sample_sd"] for item in stats],
                          color=color, marker=marker, capsize=3, label=model.upper())
        axis.set(xticks=x, xticklabels=["Clean", *[f"{snr} dB" for snr in levels]],
                 ylabel=f"{label} (%)", xlabel="Test input")
        axis.grid(axis="y", alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="outside lower center", ncols=3)
    fig.suptitle("UCI HAR official test scores · mean ± sample SD across 3 seeds")
    fig.savefig(output / "noise_scores.png", dpi=180)
    plt.close(fig)


def plot_drops(output, rows, summary, levels):
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.3), sharey=True, layout="constrained")
    x = np.arange(len(levels))
    for axis, metric, label in zip(axes, METRICS, LABELS):
        axis.axhline(0, color="#666666", linewidth=1, linestyle="--")
        for model, color, marker, offset in zip(MODELS, COLORS, MARKERS, (-0.18, 0, 0.18)):
            stats = [summary[model]["noisy"][str(level)][metric]["paired_drop_pp"] for level in levels]
            axis.errorbar(x + offset, [item["mean"] for item in stats], yerr=[item["sample_sd"] for item in stats],
                          color=color, marker=marker, linestyle="none", capsize=4, label=model.upper())
            for index, level in enumerate(levels):
                drops = [row[f"{metric}_drop_pp"] for row in rows if row["model"] == model and row["snr_db"] == level]
                axis.scatter(index + offset + np.array([-0.035, 0, 0.035]), drops, color=color,
                             marker=marker, s=13, alpha=0.35)
        axis.set(xlim=(-0.4, len(levels) - 0.6), xticks=x, xticklabels=[f"{snr} dB" for snr in levels],
                 ylabel=f"{label} drop (percentage points)", xlabel="Signal-to-noise ratio")
        axis.grid(axis="y", alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="outside lower center", ncols=3)
    fig.suptitle("Paired score changes · clean minus noisy · mean ± sample SD")
    fig.savefig(output / "noise_drops.png", dpi=180)
    plt.close(fig)


def build_report(source, output):
    if output.exists():
        raise FileExistsError(f"Report already exists at {output}. Choose a new output directory.")
    data, runs, levels = read_comparison(source)
    rows, summary = summarize(data, runs, levels)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".noise-report-", dir=output.parent) as temporary:
        staged = Path(temporary) / "report"
        staged.mkdir()
        with (staged / "per_run.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        metadata = {key: data[key] for key in (
            "freeze_sha256", "protocol_sha256", "dataset_fingerprint", "official_test_sha256",
            "test_subjects", "test_windows",
        )}
        metadata.update(seeds=list(SEEDS), snr_db=levels, models=summary)
        (staged / "summary.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8", newline="\n"
        )
        shutil.copyfile(source, staged / "results.json")
        shutil.copyfile(source.with_name("protocol.json"), staged / "protocol.json")
        plot_scores(staged, summary, levels)
        plot_drops(staged, rows, summary, levels)
        staged.rename(output)
    print(f"Published {len(rows)} paired comparisons to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    build_report(args.input, args.output)


if __name__ == "__main__":
    main()
