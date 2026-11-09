"""Load already standardized data without opening the official test set by default."""

from pathlib import Path
import hashlib
import json

import numpy as np

CHANNELS = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]
LABELS = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
          "SITTING", "STANDING", "LAYING"]


def load_data(data_dir="data/processed", include_test=False):
    """Return train/val arrays [N,128,9]; test requires an explicit opt-in.

    Labels are zero-based. Statistics were fitted on training people only.
    Reading test here is intended for a final frozen-model evaluation only.
    """
    root = Path(data_dir)
    metadata_file = root / "metadata.json"
    if not metadata_file.is_file():
        raise FileNotFoundError("Prepared data missing. Run: python -m har_compare.prepare")
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    if metadata["channel_order"] != CHANNELS:
        raise ValueError("Unexpected channel order; regenerate the processed dataset.")
    splits = metadata["split_subjects"]
    names = ("train", "val", "test")
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            if set(splits[first]) & set(splits[second]):
                raise ValueError(f"Subject leakage between {first} and {second}")
    result = {"metadata": metadata}
    for split in (names if include_test else names[:2]):
        data_file = root / f"{split}.npz"
        digest = hashlib.sha256()
        with data_file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != metadata["processed_sha256"][split]:
            raise ValueError(f"Processed {split} data checksum mismatch; regenerate using prepare --force.")
        with np.load(data_file, allow_pickle=False) as archive:
            item = {key: archive[key] for key in ("x", "y", "subjects")}
        x, y, subjects = item["x"], item["y"], item["subjects"]
        if x.shape != (len(y), 128, 9) or subjects.shape != y.shape:
            raise ValueError(f"Unexpected shapes in {split}")
        if x.dtype != np.float32 or y.dtype != np.int64 or subjects.dtype != np.int64:
            raise ValueError(f"Unexpected dtypes in {split}")
        if not np.isfinite(x).all() or not np.isin(y, range(6)).all():
            raise ValueError(f"Invalid input values in {split}")
        if sorted(np.unique(subjects).tolist()) != splits[split]:
            raise ValueError(f"Subject manifest mismatch in {split}")
        result[split] = item
    return result
