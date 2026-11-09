"""Shared reproducibility, evaluation and reporting utilities."""

import hashlib
import json
import os
import platform
import random
import time
from pathlib import Path

# Set before the first CUDA operation for deterministic recurrent training.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .metrics import classification_metrics

LABEL_NAMES = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]
DATA_FINGERPRINT_SCHEMA_VERSION = 1
FINGERPRINT_FIELDS = (
    "schema_version", "archive_sha256", "split_seed", "split_subjects",
    "sample_counts", "channel_order", "labels", "shape_per_sample",
    "sample_rate_hz", "normalization",
)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA GPU")
    return torch.device(requested)


def seed_everything(seed: int, threads: int = 4) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def data_loader(split: dict, batch_size: int, device: torch.device,
                shuffle: bool = False, seed: int = 42) -> DataLoader:
    x = torch.as_tensor(np.asarray(split["x"], dtype=np.float32))
    y = torch.as_tensor(np.asarray(split["y"], dtype=np.int64))
    if x.ndim != 3 or x.shape[1:] != (128, 9) or y.shape != (len(x),) or len(x) == 0:
        raise ValueError("Expected nonempty x[N,128,9] and y[N]")
    dataset = TensorDataset(x, y)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=device.type == "cuda",
                      drop_last=False, generator=generator)


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    total_loss = 0.0
    truth, predictions = [], []
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            total_loss += float(criterion(logits, y).item())
            truth.append(y.cpu().numpy())
            predictions.append(logits.argmax(dim=1).cpu().numpy())
    metrics = classification_metrics(np.concatenate(truth), np.concatenate(predictions))
    metrics["loss"] = total_loss / metrics["n_samples"]
    return metrics


def metadata_fingerprint(metadata: dict) -> str:
    """Hash data semantics, excluding creation timestamps and NPZ container hashes.

    Repacking the same arrays or preparing them on another day must not prevent
    checkpoint reuse. The original archive, split and normalization still bind
    the checkpoint to the experiment's actual inputs.
    """
    missing = sorted(set(FINGERPRINT_FIELDS) - metadata.keys())
    if missing:
        raise ValueError(f"Data metadata lacks fingerprint fields: {missing}")
    semantic = {"fingerprint_schema_version": DATA_FINGERPRINT_SCHEMA_VERSION,
                "metadata": {key: metadata[key] for key in FINGERPRINT_FIELDS}}
    payload = json.dumps(semantic, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def environment_info(device: torch.device) -> dict:
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": str(torch.__version__), "numpy": np.__version__,
        "cuda_build": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cpu_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def inference_timing(model: nn.Module, loader: DataLoader, device: torch.device,
                     repeats: int = 30) -> dict:
    """Measure model-only inference on an already resident batch, not transfer or I/O."""
    batch, _ = next(iter(loader))
    batch = batch.to(device)
    model.eval()
    with torch.inference_mode():
        for _ in range(5):
            model(batch)
        synchronize(device)
        start = time.perf_counter()
        for _ in range(repeats):
            model(batch)
        synchronize(device)
    batch_ms = (time.perf_counter() - start) * 1000 / repeats
    return {"batch_size": len(batch), "repeats": repeats, "warmup_batches": 5,
            "batch_ms": batch_ms, "amortized_ms_per_sample": batch_ms / len(batch),
            "scope": "Model forward only, resident batch; excludes data loading and transfer. "
                     "Amortized per-sample time is not batch-size-one latency."}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def plot_results(history: list[dict], metrics: dict, output_dir: Path,
                 model_name: str, stage: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="Validation")
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Learning curves")
    axes[0].legend()
    axes[1].plot(epochs, [row["val_accuracy"] for row in history], label="Accuracy")
    axes[1].plot(epochs, [row["val_macro_f1"] for row in history], label="Macro-F1")
    axes[1].set(xlabel="Epoch", ylabel="Validation score", ylim=(0, 1), title="Validation performance")
    axes[1].legend()
    fig.suptitle(f"{model_name.upper()} | {stage} run | official test set untouched")
    fig.savefig(output_dir / "learning_curves.png", dpi=150)
    plt.close(fig)

    matrix = np.asarray(metrics["confusion_matrix"])
    fig, axis = plt.subplots(figsize=(7.6, 6.5), constrained_layout=True)
    im = axis.imshow(matrix, cmap="Blues")
    fig.colorbar(im, ax=axis, shrink=0.8, label="Window count")
    axis.set(xticks=range(6), yticks=range(6), xticklabels=LABEL_NAMES,
             yticklabels=LABEL_NAMES, xlabel="Predicted activity", ylabel="True activity",
             title=f"{model_name.upper()} validation confusion matrix ({stage})")
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(6):
        for j in range(6):
            axis.text(j, i, str(matrix[i, j]), ha="center", va="center",
                      color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)
