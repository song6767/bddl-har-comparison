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
    stage = records[0]["stage"]
    lines = ["# 三个时序模型的验证集运行记录", "",
             f"阶段：`{stage}`。所有指标均来自验证集；官方测试集未进行预测或评分。", "",
             "smoke 阶段仅证明数据、训练、保存和评估流程能够运行。单种子短训练结果不能作为最终模型排名。",
             "验证集同时用于选择最佳轮次，因此这里不是独立测试结果。", "",
             "| 模型 | 种子 | 完成轮数 | 最佳轮次 | 参数量 | 验证准确率 | 验证 Macro-F1 | 训练流程秒数 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row['model'].upper()} | {row['seed']} | {row['epochs_completed']} | "
                     f"{row['best_epoch']} | {row['parameters']:,} | {row['val_accuracy']:.2%} | "
                     f"{row['val_macro_f1']:.4f} | {row['fit_seconds']:.2f} |")
    first = records[0]
    lines.extend(["", f"设备：{first['environment']['gpu_name'] or first['environment']['device']}。",
                  f"训练集 {first['train_samples']:,} 窗，验证集 {first['val_samples']:,} 窗。",
                  "标准化统计量只使用训练集。不同集合的受试者互不重叠。", "",
                  "训练流程秒数包含每轮验证和最佳检查点写入，不包含数据加载、绘图和启动进程。",
                  "显存记录为 PyTorch 分配器的峰值，不包括驱动、上下文与其他程序；推理时间为驻留设备批次的前向计算，",
                  "不包含数据传输，按批次均摊的时间不等于单样本延迟。短训练计时仅作流程记录。", "",
                  "## 结果文件", "",
                  "- [汇总 CSV](summary.csv)", "- [汇总 JSON](summary.json)",
                  "- [数据来源和人员划分](data_manifest.json)"])
    for row in rows:
        folder = f"{row['model']}/seed_{row['seed']}"
        lines.extend([f"- {row['model'].upper()}，种子 {row['seed']}："
                      f"[完整指标]({folder}/metrics.json)、[逐轮记录]({folder}/history.csv)、"
                      f"[训练曲线]({folder}/learning_curves.png)、[混淆矩阵]({folder}/confusion_matrix.png)"])
    lines.extend(["", "## 后续实验", "",
                  "先完成相同调参预算下的验证实验，再冻结各模型配置，进行三个随机种子的正式训练。",
                  "最终统一评估官方测试集，并按预先确定的方案分析效率和噪声敏感性。", ""])
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Exported {len(records)} validation runs to {output}; datasets and weights excluded.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", default="runs/smoke")
    parser.add_argument("--output-dir", default="reports/smoke")
    args = parser.parse_args()
    summarize(args.runs_root, args.output_dir)


if __name__ == "__main__":
    main()
