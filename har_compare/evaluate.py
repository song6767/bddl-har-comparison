"""Evaluate a saved model; official test evaluation requires an explicit flag."""

import argparse
import json
from pathlib import Path

from .runtime import (DATA_FINGERPRINT_SCHEMA_VERSION, data_loader, environment_info, evaluate_model, metadata_fingerprint,
                      resolve_device, seed_everything, write_json)

import torch

from .data import load_data
from .models import build_model


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--checkpoint", required=True)
    result.add_argument("--data-dir", default="data/processed")
    result.add_argument("--split", choices=["val", "test"], default="val")
    result.add_argument("--allow-test", action="store_true",
                        help="Evaluate official test only after model choices and protocol are frozen")
    result.add_argument("--batch-size", type=int, default=64)
    result.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    result.add_argument("--output", help="Optional destination for evaluation JSON")
    return result


def evaluate(args: argparse.Namespace) -> dict:
    if args.split == "test" and not args.allow_test:
        raise ValueError("Official test evaluation is sealed by default. Freeze model choices first, "
                         "then explicitly pass --split test --allow-test.")
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive")
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if checkpoint.get("data_fingerprint_schema_version") != DATA_FINGERPRINT_SCHEMA_VERSION:
        raise ValueError("Checkpoint data fingerprint schema is unsupported")
    seed_everything(checkpoint["config"]["seed"], checkpoint["config"].get("threads", 4))
    dataset = load_data(args.data_dir, include_test=args.split == "test")
    if metadata_fingerprint(dataset["metadata"]) != checkpoint["data_metadata_sha256"]:
        raise ValueError("Prepared data metadata differs from the checkpoint. Use the original split and normalization.")
    model = build_model(checkpoint["model_name"], dropout=checkpoint["config"].get("dropout", 0.2)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = evaluate_model(model, data_loader(dataset[args.split], args.batch_size, device), device)
    report = {"model": checkpoint["model_name"], "split": args.split,
              "stage": checkpoint["config"].get("stage", "unknown"),
              "checkpoint_epoch": checkpoint["epoch"], "metrics": metrics,
              "environment": environment_info(device),
              "data_metadata_sha256": checkpoint["data_metadata_sha256"],
              "data_fingerprint_schema_version": DATA_FINGERPRINT_SCHEMA_VERSION,
              "test_evaluated": args.split == "test"}
    if args.output:
        write_json(Path(args.output), report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main() -> None:
    evaluate(parser().parse_args())


if __name__ == "__main__":
    main()
