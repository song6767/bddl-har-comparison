"""Export the validation search and the frozen multi-seed test comparison."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from .formal_train import FORMAL_SEEDS as SEEDS, LEARNING_RATES, MODELS, read_selection
from .runtime import write_json


CLASS_NAMES = ("Walk", "Upstairs", "Downstairs", "Sit", "Stand", "Lie")
CLASS_NAMES_ZH = ("步行", "上楼", "下楼", "坐姿", "站姿", "躺卧")
COLORS = ("#0072B2", "#D55E00", "#009E73")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def describe(values):
    if all(value is None for value in values):
        return None
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "sample_sd": float(values.std(ddof=1))}


def collect(root, protocol):
    """Check that the published tables contain all nine paired runs."""
    selection = read_json(root / "selection.json")
    freeze_path = root / "freeze.json"
    freeze = read_json(freeze_path)
    freeze_hash = digest(freeze_path)
    if (digest(root / "protocol.json") != freeze["protocol_sha256"]
            or digest(root / "selection.json") != freeze["selection_sha256"]):
        raise ValueError("Protocol or selection differs from the frozen record")
    if selection != read_selection(root, protocol):
        raise ValueError("The frozen learning-rate selection differs from the validation search")
    frozen = {(row["model"], row["seed"]): row for row in freeze["checkpoints"]}
    if len(freeze["checkpoints"]) != len(MODELS) * len(SEEDS) or set(frozen) != {
            (model, seed) for model in MODELS for seed in SEEDS}:
        raise ValueError("Freeze manifest must contain the nine expected checkpoints")
    rates = selection["learning_rates"]
    tuning, rows, histories, tests, matrices = [], [], {}, {}, {}
    fingerprints = set()
    for model in MODELS:
        for rate in LEARNING_RATES:
            pilot = read_json(root / "tuning" / model / f"lr_{rate:g}" / "metrics.json")
            fingerprints.add(pilot["data_metadata_sha256"])
            tuning.append({"model": model, "learning_rate": rate, "seed": pilot["seed"],
                           "selected": rate == rates[model], "best_epoch": pilot["best_epoch"],
                           "epochs_completed": pilot["epochs_completed"], "val_accuracy": pilot["val_accuracy"],
                           "val_macro_f1": pilot["val_macro_f1"]})

        matrices[model] = np.zeros((6, 6), dtype=np.int64)
        for seed in SEEDS:
            train_dir = root / "training" / model / f"seed_{seed}"
            train = read_json(train_dir / "metrics.json")
            test = read_json(root / "test" / model / f"seed_{seed}.json")
            if (train["model"], train["stage"], train["seed"], train["config"]["lr"], train["test_evaluated"]) != (
                    model, "formal", seed, rates[model], False):
                raise ValueError(f"Unexpected training metadata for {model}, seed {seed}")
            if (test["model"], test["stage"], test["seed"], test["split"], test["test_evaluated"],
                    test["checkpoint_epoch"]) != (model, "formal", seed, "test", True, train["best_epoch"]):
                raise ValueError(f"Unexpected evaluation metadata for {model}, seed {seed}")
            entry = frozen[(model, seed)]
            if (test["freeze_sha256"] != freeze_hash or test["checkpoint_sha256"] != entry["sha256"]
                    or digest(train_dir / "metrics.json") != entry["metrics_sha256"]):
                raise ValueError(f"Evaluation does not match the frozen checkpoint for {model}, seed {seed}")
            fingerprints.update([train["data_metadata_sha256"], test["data_metadata_sha256"]])
            with (train_dir / "history.csv").open(newline="", encoding="utf-8") as stream:
                history = list(csv.DictReader(stream))
            if len(history) != train["epochs_completed"]:
                raise ValueError(f"Incomplete training history for {model}, seed {seed}")
            counts = np.asarray(test["metrics"]["confusion_matrix"], dtype=np.int64)
            if counts.shape != (6, 6) or (counts < 0).any() or counts.sum() != test["metrics"]["n_samples"]:
                raise ValueError(f"Invalid test confusion matrix for {model}, seed {seed}")
            matrices[model] += counts
            rows.append({"model": model, "seed": seed, "learning_rate": rates[model],
                         "best_epoch": train["best_epoch"], "epochs_completed": train["epochs_completed"],
                         "parameters": train["parameters"], "val_accuracy": train["val_accuracy"],
                         "val_macro_f1": train["val_macro_f1"], "test_accuracy": test["metrics"]["accuracy"],
                         "test_macro_f1": test["metrics"]["macro_f1"], "test_samples": test["metrics"]["n_samples"],
                         "fit_seconds": train["fit_seconds"], "fit_seconds_per_epoch": train["fit_seconds"] / len(history),
                         "peak_torch_allocated_mib": train["peak_torch_allocated_mib"],
                         "inference_batch_ms": train["inference"]["batch_ms"],
                         "checkpoint_sha256": test["checkpoint_sha256"]})
            histories[(model, seed)] = history
            tests[(model, seed)] = test
    if fingerprints != {freeze["dataset_fingerprint"]} or len({row["test_samples"] for row in rows}) != 1:
        raise ValueError("Training and evaluation runs must use the same dataset and splits")
    return tuning, rows, histories, tests, matrices, fingerprints.pop()


def plot_scores(output, rows, summary):
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.1), constrained_layout=True)
    x = np.arange(len(MODELS))
    for axis, key, ylabel in zip(axes, ("test_accuracy", "test_macro_f1"), ("Accuracy (%)", "Macro-F1 (%)")):
        for seed, offset in zip(SEEDS, (-0.12, 0, 0.12)):
            values = [row[key] * 100 for row in rows if row["seed"] == seed]
            axis.plot(x + offset, values, "o-", linewidth=1, markersize=4.5, alpha=0.72, label=f"Seed {seed}")
        means = [100 * summary[model][key]["mean"] for model in MODELS]
        errors = [100 * summary[model][key]["sample_sd"] for model in MODELS]
        axis.errorbar(x + 0.23, means, yerr=errors, fmt="D", color="#222222", capsize=3,
                      markersize=4.5, label="Mean ± sample SD")
        axis.set(xticks=x, xticklabels=[model.upper() for model in MODELS], ylabel=ylabel)
        axis.set_xlim(-0.35, len(MODELS) - 0.58)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("UCI HAR official test set · three paired training seeds", fontsize=13)
    fig.savefig(output / "test_scores.png", dpi=180)
    plt.close(fig)


def plot_confusion(output, matrices):
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.1), constrained_layout=True)
    for axis, model in zip(axes, MODELS):
        matrix = matrices[model]
        percentages = 100 * matrix / matrix.sum(axis=1, keepdims=True)
        plotted = axis.imshow(percentages, cmap="Blues", vmin=0, vmax=100)
        for (i, j), value in np.ndenumerate(percentages):
            axis.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=8,
                      color="white" if value > 55 else "#163049")
        axis.set(title=model.upper(), xticks=range(6), yticks=range(6),
                 xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, xlabel="Predicted activity")
        plt.setp(axis.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    axes[0].set_ylabel("True activity")
    fig.colorbar(plotted, ax=axes, shrink=0.82, label="% within true class")
    fig.suptitle("Official test confusion matrices · aggregated across three runs", fontsize=13)
    fig.savefig(output / "test_confusion.png", dpi=180)
    plt.close(fig)


def plot_learning_curves(output, histories):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7), sharex=True, sharey=True, constrained_layout=True)
    for axis, model in zip(axes, MODELS):
        for seed, color in zip(SEEDS, COLORS):
            rows = histories[(model, seed)]
            axis.plot([int(row["epoch"]) for row in rows], [100 * float(row["val_macro_f1"]) for row in rows],
                      color=color, linewidth=1.2, label=f"Seed {seed}")
        axis.set(title=model.upper(), xlabel="Epoch")
        axis.grid(axis="y", alpha=0.23)
    axes[0].set_ylabel("Validation Macro-F1 (%)")
    axes[0].xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    axes[2].legend(frameon=False, loc="lower right", fontsize=8)
    fig.suptitle("Formal training with validation-based early stopping", fontsize=13)
    fig.savefig(output / "validation_curves.png", dpi=180)
    plt.close(fig)


def plot_efficiency(output, summary, device_name, batch_size):
    descriptors = [("fit_seconds_per_epoch", "Fit time per epoch (s)"),
                   ("inference_batch_ms", "Forward time per batch (ms)")]
    if summary[MODELS[0]]["peak_torch_allocated_mib"] is not None:
        descriptors.append(("peak_torch_allocated_mib", "Peak allocated GPU memory (MiB)"))
    fig, axes = plt.subplots(1, len(descriptors), figsize=(3.5 * len(descriptors), 3.5), constrained_layout=True)
    for axis, (key, ylabel) in zip(axes, descriptors):
        values = [summary[model][key] for model in MODELS]
        axis.bar([model.upper() for model in MODELS], [item["mean"] for item in values],
                 yerr=[item["sample_sd"] for item in values], color=COLORS, capsize=3, width=0.6)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.23)
        axis.set_axisbelow(True)
    fig.suptitle(f"{device_name} efficiency · batch size {batch_size} · mean ± sample SD", fontsize=13)
    fig.savefig(output / "efficiency.png", dpi=180)
    plt.close(fig)


def format_stats(values, scale=1, decimals=2):
    if values is None:
        return "—"
    return (f"{scale * values['mean']:.{decimals}f} ± "
            f"{scale * values['sample_sd']:.{decimals}f}")


def major_confusion(matrix):
    # Count a pair in both directions. Report against all incorrect predictions.
    pairs = np.triu(matrix + matrix.T, k=1)
    i, j = np.unravel_index(pairs.argmax(), pairs.shape)
    errors = int(matrix.sum() - matrix.trace())
    return {"classes": [int(i), int(j)], "count": int(pairs[i, j]), "total_errors": errors}


def write_readme(output, tuning, rows, summary, confusions, manifest, protocol, device_name):
    ranked = sorted(MODELS, key=lambda model: summary[model]["test_macro_f1"]["mean"], reverse=True)
    first, second = ranked[:2]
    gap = 100 * (summary[first]["test_macro_f1"]["mean"] - summary[second]["test_macro_f1"]["mean"])
    first_wins = sum(max((row for row in rows if row["seed"] == seed), key=lambda row: row["test_macro_f1"])["model"]
                     == first for seed in SEEDS)
    lines = ["# CNN RNN 和 LSTM 的正式比较", "",
             f"在 UCI HAR 官方测试集上，{first.upper()} 的三种子平均 Macro-F1 最高，"
             f"比第二位的 {second.upper()} 高 {gap:.2f} 个百分点。"
             f"{first.upper()} 在 {first_wins}/{len(SEEDS)} 个种子上的 Macro-F1 最高。", "",
             "## 数据和模型选择", ""]
    sizes, people = manifest["sample_counts"], manifest["split_subjects"]
    lines.append(f"训练、验证和官方测试分别使用 {len(people['train'])}、{len(people['val'])} 和 "
                 f"{len(people['test'])} 名互不重叠的受试者，对应 {sizes['train']:,}、{sizes['val']:,} 和 "
                 f"{sizes['test']:,} 个窗口。每个窗口有 128 个时间点和 9 个信号通道。标准化统计量只用训练集估计。")
    lines.extend(["", f"每个模型用同一个种子 {protocol['tuning_seed']} 尝试三个学习率。训练最多 {protocol['max_epochs']} 轮，"
                  f"按验证 Macro-F1 选出最优轮次，连续 {protocol['patience']} 轮无提升则早停。"
                  "验证分数相同则选择较小的学习率。三个模型采用相同批量大小、优化器和梯度裁剪阈值。选定学习率后使用种子 "
                  "42、43、44 各自重新训练。9 个检查点全部记录哈希后，再统一评估官方测试集。", "",
                  "| 模型 | 学习率 | 验证准确率 | 验证 Macro-F1 | 最佳轮 / 完成轮 | 选中 |",
                  "| --- | ---: | ---: | ---: | ---: | --- |"])
    for row in tuning:
        lines.append(f"| {row['model'].upper()} | {row['learning_rate']:g} | {row['val_accuracy']:.2%} | "
                     f"{row['val_macro_f1']:.4f} | {row['best_epoch']} / {row['epochs_completed']} | "
                     f"{'是' if row['selected'] else ''} |")
    largest_rate = max(LEARNING_RATES)
    if all(entry["learning_rate"] == largest_rate for entry in summary.values()):
        lines.extend(["", f"三个模型均选中候选范围的上边界 `{largest_rate:g}`，学习率结论限于本次搜索网格。"])
    lines.extend(["", "## 官方测试集", "",
                  "表中为 3 次正式训练的均值 ± 样本标准差。准确率和 Macro-F1 均用百分数表示。", "",
                  "| 模型 | 学习率 | 验证 Macro-F1 (%) | 测试准确率 (%) | 测试 Macro-F1 (%) |",
                  "| --- | ---: | ---: | ---: | ---: |"])
    for model in MODELS:
        entry = summary[model]
        lines.append(f"| {model.upper()} | {entry['learning_rate']:g} | {format_stats(entry['val_macro_f1'], 100)} | "
                     f"{format_stats(entry['test_accuracy'], 100)} | {format_stats(entry['test_macro_f1'], 100)} |")
    val_best = max(MODELS, key=lambda model: summary[model]["val_macro_f1"]["mean"])
    lines.extend(["", "![三种子官方测试指标](test_scores.png)", "",
                  "三次评估共用同一批测试受试者，标准差描述训练随机性。窗口有重叠，受试者也只有 9 名，"
                  "因此这些结果的适用范围限于该数据集的固定划分和采集条件。", ""])
    if val_best != first:
        lines.extend([f"{val_best.upper()} 的平均验证 Macro-F1 最高，{first.upper()} 的平均测试 Macro-F1 最高。"
                      "验证和测试来自不同受试者，且验证分数参与了轮次选择。", ""])
    lines.extend(["## 主要混淆", "", "![官方测试集混淆矩阵](test_confusion.png)", "",
                  "矩阵先将三个种子的计数相加，再按真实类别归一化。每个测试窗口重复计入三次。各模型最常见的双向混淆如下。", ""])
    for model in MODELS:
        item = confusions[model]
        if item["total_errors"]:
            names = "与".join(CLASS_NAMES_ZH[i] for i in item["classes"])
            lines.append(f"- {model.upper()} 的{names}混淆合计 {item['count']} 次，"
                         f"占该模型 {item['total_errors']} 次错误的 {item['count'] / item['total_errors']:.1%}。")
    batch_size = protocol["batch_size"]
    lines.extend(["", "## 计算效率", "",
                  f"三个模型均在 {device_name} 上顺序运行。参数量来自模型本身，其余指标为三个种子的均值 ± 样本标准差。", "",
                  f"| 模型 | 参数量 | 完成轮数 | 总训练秒数 | 每轮秒数 | {batch_size} 样本前向毫秒 | 峰值分配显存 MiB |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for model in MODELS:
        entry = summary[model]
        lines.append(f"| {model.upper()} | {entry['parameters']:,} | {format_stats(entry['epochs_completed'], decimals=1)} | "
                     f"{format_stats(entry['fit_seconds'])} | {format_stats(entry['fit_seconds_per_epoch'], decimals=3)} | "
                     f"{format_stats(entry['inference_batch_ms'], decimals=3)} | "
                     f"{format_stats(entry['peak_torch_allocated_mib'], decimals=1)} |")
    fastest = min(MODELS, key=lambda model: summary[model]["fit_seconds_per_epoch"]["mean"])
    if fastest != first:
        ratio = summary[first]["fit_seconds_per_epoch"]["mean"] / summary[fastest]["fit_seconds_per_epoch"]["mean"]
        lines.extend(["", f"{fastest.upper()} 的平均每轮训练耗时最短。测试 Macro-F1 最高的 {first.upper()} 每轮耗时约为其 {ratio:.2f} 倍。"])
    else:
        next_fastest = min((model for model in MODELS if model != fastest),
                           key=lambda model: summary[model]["fit_seconds_per_epoch"]["mean"])
        reduction = 1 - summary[fastest]["fit_seconds_per_epoch"]["mean"] / summary[next_fastest]["fit_seconds_per_epoch"]["mean"]
        lines.extend(["", f"{fastest.upper()} 的平均每轮训练耗时最短，比第二快的 {next_fastest.upper()} 少约 {reduction:.0%}。"])
    lines.extend(["", "![每轮训练耗时、批推理耗时与峰值显存](efficiency.png)", "",
                  "总训练时间包含每轮验证与最佳权重保存。每轮秒数按各次训练的总耗时除以完成轮数。"
                  f"批前向从验证集取 {batch_size} 个样本，放入运行设备后预热 5 次并计时 30 次。"
                  "CUDA 计时使用同步，峰值显存采用 PyTorch 分配器统计。CPU 运行的显存项记为 —。", "",
                  "![正式训练的验证集曲线](validation_curves.png)", "",
                  "## 复核材料", "",
                  "- [逐次正式运行与测试结果](per_run.csv)", "- [完整学习率搜索表](tuning.csv)",
                  "- [机器可读汇总](summary.json)", "- [预设实验配置](protocol.json)",
                  "- [配置选择记录](selection.json)", "- [正式检查点哈希及冻结记录](freeze.json)",
                  "- [数据划分与标准化信息](data_manifest.json)",
                  "- [正式训练历史](training/) · [测试明细](test/) · [调参原始记录](tuning/)", ""])
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def build_report(root, output):
    protocol = read_json(root / "protocol.json")
    tuning, rows, histories, tests, matrices, fingerprint = collect(root, protocol)
    config = read_json(root / "training" / "cnn" / "seed_42" / "config.json")
    manifest = config["data_metadata"]
    device_name = config["environment"]["gpu_name"] or config["environment"]["device"].upper()
    if rows[0]["test_samples"] != manifest["sample_counts"]["test"]:
        raise ValueError("Test evaluation window count does not match the data manifest")
    summary = {}
    numeric_fields = ("val_accuracy", "val_macro_f1", "test_accuracy", "test_macro_f1", "epochs_completed",
                      "fit_seconds", "fit_seconds_per_epoch", "inference_batch_ms", "peak_torch_allocated_mib")
    for model in MODELS:
        matching = [row for row in rows if row["model"] == model]
        summary[model] = {"learning_rate": matching[0]["learning_rate"], "parameters": matching[0]["parameters"]}
        summary[model].update({key: describe([row[key] for row in matching]) for key in numeric_fields})

    output.mkdir(parents=True, exist_ok=True)
    for filename in ("protocol.json", "selection.json", "freeze.json"):
        shutil.copyfile(root / filename, output / filename)
    write_csv(output / "tuning.csv", tuning)
    write_csv(output / "per_run.csv", rows)
    write_json(output / "data_manifest.json", manifest)
    confusions = {model: major_confusion(matrix) for model, matrix in matrices.items()}
    write_json(output / "summary.json", {"test_evaluated": True, "data_metadata_sha256": fingerprint,
               "tuning_seed": protocol["tuning_seed"], "formal_seeds": list(SEEDS), "test_windows_per_seed": rows[0]["test_samples"],
               "test_subjects": len(manifest["split_subjects"]["test"]), "models": summary,
               "environment": config["environment"], "test_labels": manifest["labels"],
               "confusion_count_scope": "Same test windows scored by each of three checkpoints per model",
               "test_confusion_counts_sum_across_seeds": {model: value.tolist() for model, value in matrices.items()},
               "major_confusions": confusions})
    # Mirror the experiment's relative paths so the published files can be
    # matched directly to the selection and freeze manifest hashes.
    paths = []
    for model, seed in tests:
        training = Path("training") / model / f"seed_{seed}"
        paths.extend(training / filename for filename in ("history.csv", "metrics.json", "config.json"))
        paths.append(Path("test") / model / f"seed_{seed}.json")
    for row in tuning:
        rate = row["learning_rate"]
        paths.append(Path("tuning") / row["model"] / f"lr_{rate:g}" / "metrics.json")
    for relative_path in paths:
        destination = output / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative_path, destination)
    plot_scores(output, rows, summary)
    plot_confusion(output, matrices)
    plot_learning_curves(output, histories)
    plot_efficiency(output, summary, device_name, protocol["batch_size"])
    write_readme(output, tuning, rows, summary, confusions, manifest, protocol, device_name)
    print(f"Published {len(tuning)} tuning trials and {len(rows)} formal evaluations to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/formal_v1"))
    parser.add_argument("--output", type=Path, default=Path("reports/formal"))
    args = parser.parse_args()
    build_report(args.root, args.output)


if __name__ == "__main__":
    main()
