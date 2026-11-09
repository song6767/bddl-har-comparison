"""Train one model using training subjects and validation subjects only."""

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import time

from .runtime import (DATA_FINGERPRINT_SCHEMA_VERSION, data_loader, environment_info, evaluate_model, inference_timing,
                      metadata_fingerprint, plot_results, resolve_device,
                      seed_everything, synchronize, write_json)

import numpy as np
import torch
from torch import nn

from .data import load_data
from .models import build_model, count_parameters


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model", choices=["cnn", "rnn", "lstm"], required=True)
    result.add_argument("--data-dir", default="data/processed")
    result.add_argument("--output-dir", required=True)
    result.add_argument("--epochs", type=int, default=50)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--batch-size", type=int, default=64)
    result.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    result.add_argument("--lr", type=float, default=1e-3)
    result.add_argument("--patience", type=int, default=8)
    result.add_argument("--dropout", type=float, default=0.2)
    result.add_argument("--threads", type=int, default=4)
    result.add_argument("--stage", choices=["smoke", "pilot", "formal"], default="pilot")
    result.add_argument("--overwrite", action="store_true", help="Replace artifacts in an existing run directory")
    return result


def train(args: argparse.Namespace) -> dict:
    if min(args.epochs, args.batch_size, args.patience, args.threads) < 1 or args.lr <= 0:
        raise ValueError("Epochs, batch size, patience, threads and learning rate must be positive")
    if args.seed < 0 or args.seed >= 2**32:
        raise ValueError("Seed must be an integer in [0, 2**32)")
    output_dir = Path(args.output_dir)
    if not args.overwrite and any((output_dir / name).exists() for name in ("best.pt", "metrics.json", "history.csv")):
        raise FileExistsError(f"Run already exists at {output_dir}; choose a new directory or pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed, args.threads)
    device = resolve_device(args.device)
    # The default data API does not open the official test array.
    dataset = load_data(args.data_dir)
    training_loader = data_loader(dataset["train"], args.batch_size, device, shuffle=True, seed=args.seed)
    validation_loader = data_loader(dataset["val"], args.batch_size, device, seed=args.seed)
    model = build_model(args.model, dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    data_metadata = dataset["metadata"]
    fingerprint = metadata_fingerprint(data_metadata)
    config = vars(args).copy()
    config.update({"gradient_clip_norm": 1.0, "optimizer": "Adam", "selection_metric": "val_macro_f1",
                   "test_evaluated": False, "model_input_shape": [128, 9],
                   "pooling": "temporal mean", "num_workers": 0,
                   "parameters": count_parameters(model)})
    environment = environment_info(device)
    started_at = datetime.now(timezone.utc).isoformat()
    write_json(output_dir / "config.json", {"config": config, "environment": environment,
               "started_at_utc": started_at, "data_metadata": data_metadata,
               "data_fingerprint_schema_version": DATA_FINGERPRINT_SCHEMA_VERSION,
               "data_metadata_sha256": fingerprint})
    print(f"{args.model.upper()} | {args.stage} | {device} | parameters={config['parameters']:,} | "
          f"train={len(training_loader.dataset)} val={len(validation_loader.dataset)} | official test untouched", flush=True)

    best_score, best_epoch, stale_epochs = -1.0, 0, 0
    history = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    fit_start = time.perf_counter()
    fields = ["epoch", "train_loss", "val_loss", "val_accuracy", "val_macro_f1", "epoch_seconds"]
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as history_file:
        writer = csv.DictWriter(history_file, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            synchronize(device)
            epoch_start = time.perf_counter()
            model.train()
            loss_sum = 0.0
            for x, y in training_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x), y)
                if not torch.isfinite(loss).item():
                    raise RuntimeError(f"Non-finite training loss at epoch {epoch}")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
                optimizer.step()
                loss_sum += float(loss.item()) * len(y)
            validation = evaluate_model(model, validation_loader, device)
            synchronize(device)
            row = {"epoch": epoch, "train_loss": loss_sum / len(training_loader.dataset),
                   "val_loss": validation["loss"], "val_accuracy": validation["accuracy"],
                   "val_macro_f1": validation["macro_f1"],
                   "epoch_seconds": time.perf_counter() - epoch_start}
            if not all(np.isfinite(row[key]) for key in fields if key != "epoch"):
                raise RuntimeError(f"Non-finite metric at epoch {epoch}")
            history.append(row)
            writer.writerow(row)
            history_file.flush()
            improved = validation["macro_f1"] > best_score
            if improved:
                best_score, best_epoch, stale_epochs = validation["macro_f1"], epoch, 0
                torch.save({
                    "format_version": 1, "model_name": args.model, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(), "config": config,
                    "epoch": epoch, "best_val_macro_f1": best_score,
                    "validation": validation, "environment": environment,
                    "data_metadata": data_metadata, "data_metadata_sha256": fingerprint,
                    "data_fingerprint_schema_version": DATA_FINGERPRINT_SCHEMA_VERSION,
                    "started_at_utc": started_at,
                }, output_dir / "best.pt")
            else:
                stale_epochs += 1
            print(f"epoch {epoch:02d}/{args.epochs} | train_loss={row['train_loss']:.4f} | "
                  f"val_acc={validation['accuracy']:.4f} | val_macro_f1={validation['macro_f1']:.4f} | "
                  f"{row['epoch_seconds']:.2f}s{' | best' if improved else ''}", flush=True)
            if stale_epochs >= args.patience:
                print(f"Early stopping after {stale_epochs} epochs without validation Macro-F1 improvement.", flush=True)
                break
    synchronize(device)
    fit_seconds = time.perf_counter() - fit_start
    peak_allocated = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
    peak_reserved = torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else None

    checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    validation = evaluate_model(model, validation_loader, device)
    timing = inference_timing(model, validation_loader, device)
    metrics = {
        "status": "completed", "stage": args.stage, "model": args.model, "seed": args.seed,
        "best_epoch": best_epoch, "epochs_completed": len(history), "epochs_requested": args.epochs,
        "parameters": count_parameters(model), "val_accuracy": validation["accuracy"],
        "val_macro_f1": validation["macro_f1"], "validation": validation,
        "train_samples": len(training_loader.dataset), "val_samples": len(validation_loader.dataset),
        "train_subjects": sorted(np.unique(dataset["train"]["subjects"]).astype(int).tolist()),
        "val_subjects": sorted(np.unique(dataset["val"]["subjects"]).astype(int).tolist()),
        "test_evaluated": False, "fit_seconds": fit_seconds,
        "fit_time_scope": "Training, per-epoch validation and best checkpoint writes; excludes data loading and plotting.",
        "peak_torch_allocated_mib": peak_allocated, "peak_torch_reserved_mib": peak_reserved,
        "memory_scope": "PyTorch CUDA allocator peak during fitting; excludes driver/context and other applications.",
        "inference": timing, "environment": environment, "config": config,
        "data_metadata_sha256": fingerprint,
        "data_fingerprint_schema_version": DATA_FINGERPRINT_SCHEMA_VERSION, "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation": "Validation result used for model selection; not an independent test result. "
                          "Smoke runs verify the pipeline and do not establish a model ranking.",
    }
    write_json(output_dir / "metrics.json", metrics)
    plot_results(history, validation, output_dir, args.model, args.stage)
    print(f"Completed: best epoch={best_epoch}, val_accuracy={validation['accuracy']:.4f}, "
          f"val_macro_f1={validation['macro_f1']:.4f}; artifacts: {output_dir}", flush=True)
    return metrics


def main() -> None:
    train(parser().parse_args())


if __name__ == "__main__":
    main()
