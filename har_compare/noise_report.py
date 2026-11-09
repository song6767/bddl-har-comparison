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
ACTIVITIES_ZH = ("走路", "上楼", "下楼", "坐姿", "站姿", "躺卧")


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


def fmt(stats, signed=False):
    value = f"{stats['mean']:+.2f}" if signed else f"{stats['mean']:.2f}"
    return f"{value} ± {stats['sample_sd']:.2f}"


def findings(summary, levels):
    ranks = {}
    for metric in METRICS:
        ranks[metric] = [
            sorted(MODELS, key=lambda model: summary[model]["clean"][metric]["mean"], reverse=True)
        ] + [
            sorted(MODELS, key=lambda model: summary[model]["noisy"][str(level)][metric]["score_pct"]["mean"],
                   reverse=True) for level in levels
        ]
    notes = []
    if ranks["accuracy"] == ranks["macro_f1"] and all(rank == ranks["accuracy"][0] for rank in ranks["accuracy"]):
        ranking = "、".join(model.upper() for model in ranks["accuracy"][0])
        notes.append(f"准确率与 Macro-F1 的均值在三种条件下均按 {ranking} 从高到低排列。")
    stronger = min(levels)
    f1 = {model: summary[model]["noisy"][str(stronger)]["macro_f1"]["paired_drop_pp"] for model in MODELS}
    best = min(f1, key=lambda model: f1[model]["mean"])
    worst = max(f1, key=lambda model: f1[model]["mean"])
    notes.append(f"{stronger} dB 下，{best.upper()} 的平均 Macro-F1 降幅最小"
                 f"（{f1[best]['mean']:+.2f} 个百分点），{worst.upper()} 最大（{f1[worst]['mean']:+.2f}）。")
    if all(value["sample_sd"] > abs(value["mean"]) for value in f1.values()):
        notes[-1] += "三个模型的种子间降幅标准差均大于各自的平均降幅，模型间的鲁棒性差异应按描述性结果解读。"

    clean = np.asarray(summary["cnn"]["confusion_counts_summed_across_seeds"]["clean"])
    noisy = np.asarray(summary["cnn"]["confusion_counts_summed_across_seeds"][str(stronger)])
    recalls_drop = 100 * (clean.diagonal() / clean.sum(axis=1) - noisy.diagonal() / noisy.sum(axis=1))
    activity = int(np.argmax(recalls_drop))
    accuracy_drop = summary["cnn"]["noisy"][str(stronger)]["accuracy"]["paired_drop_pp"]["mean"]
    if recalls_drop[activity] > max(abs(accuracy_drop), 0.1):
        notes.append(f"CNN 在 {stronger} dB 下的整体准确率平均变化为 {-accuracy_drop:+.2f} 个百分点，"
                     f"但合并三个种子后，{ACTIVITIES_ZH[activity]}的召回率下降 {recalls_drop[activity]:.2f} 个百分点。"
                     "整体指标掩盖了部分活动类别的变化。")
    return notes


def write_readme(output, data, levels, summary):
    protocol = data["protocol"]
    lines = [
        "# 冻结模型的人工噪声鲁棒性实验", "",
        f"使用原有 9 个冻结检查点，在 UCI HAR 官方测试集的 {data['test_subjects']} 名受试者、"
        f"{data['test_windows']:,} 个窗口上评估。重新计算的干净输入结果已与原正式结果核对。"
        "[干净输入正式实验](../formal/README.md)", "",
        f"用 {protocol['generator']} 和固定噪声种子 `{protocol['noise_seed']}` 生成一个高斯噪声模板。"
        "在标准化后的输入空间内，对每个窗口的每个通道沿 128 个时间点计算均方功率（含直流分量），"
        "按目标信噪比缩放噪声。零功率通道保持原值。两档共用同一噪声模板，各档输入由全部模型和训练种子共用。"
        "加噪后的输入转成 float32，无裁剪和二次标准化。", "",
        "## 分数和配对降幅", "",
        "数值为三个种子的均值 ± 样本标准差。分数用百分数表示。降幅按同一检查点的干净分数减去带噪分数，单位为百分点。"
        "负值表示在这一噪声实现下，平均分数略有上升。", "",
        "| 模型 | 条件 | 准确率（%） | Macro-F1（%） | 准确率降幅（百分点） | Macro-F1 降幅（百分点） |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for model in MODELS:
        item = summary[model]
        clean = [fmt(item["clean"][metric]) for metric in METRICS]
        lines.append(f"| {model.upper()} | 干净输入 | {clean[0]} | {clean[1]} | — | — |")
        for snr in levels:
            noisy = item["noisy"][str(snr)]
            scores = [fmt(noisy[metric]["score_pct"]) for metric in METRICS]
            drops = [fmt(noisy[metric]["paired_drop_pp"], signed=True) for metric in METRICS]
            lines.append(f"| {model.upper()} | {snr} dB | {scores[0]} | {scores[1]} | {drops[0]} | {drops[1]} |")
    lines.extend(["", "![不同噪声条件下的准确率和 Macro-F1](noise_scores.png)", "",
                  "![配对分数降幅。浅色点是单个训练种子的降幅，误差线表示样本标准差。](noise_drops.png)",
                  "", "## 观察与适用范围", ""])
    lines.extend(f"- {note}" for note in findings(summary, levels))
    lines.extend([
        "", "测试受试者和噪声实现固定，标准差只描述训练种子之间的变化。"
        "结果对应这两档窗口级人工加性噪声，推广到其他噪声实现、受试者或真实传感器干扰仍需额外实验。", "",
        "## 复核材料", "",
        "- [逐检查点分数和配对降幅](per_run.csv)",
        "- [模型汇总与混淆矩阵计数](summary.json)",
        "- [原始评估记录及校验值](results.json)",
        "- [预先固定的噪声协议](protocol.json)", "",
    ])
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


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
        write_readme(staged, data, levels, summary)
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
