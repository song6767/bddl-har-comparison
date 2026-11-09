"""Download the official UCI HAR archive and prepare subject-disjoint sequences."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import requests

from .data import CHANNELS, LABELS, load_data

SOURCE_URL = ("https://archive.ics.uci.edu/static/public/240/"
              "human%2Bactivity%2Brecognition%2Busing%2Bsmartphones.zip")
SOURCE_PAGE = "https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_archive(path):
    """Stream to a temporary file and publish only a complete valid ZIP."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        with zipfile.ZipFile(path) as archive:
            bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"Corrupt cached archive entry: {bad}; remove the archive and retry.")
        print(f"Using cached archive: {path}", flush=True)
        return
    temporary = path.with_suffix(".zip.part")
    for attempt in range(1, 4):
        try:
            print(f"Downloading official UCI archive (attempt {attempt})...", flush=True)
            with requests.get(SOURCE_URL, stream=True, timeout=(20, 60)) as response:
                response.raise_for_status()
                written = 0
                next_notice = 10 * 1024 * 1024
                with temporary.open("wb") as stream:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            stream.write(chunk)
                            written += len(chunk)
                            if written >= next_notice:
                                print(f"  {written / 1024**2:.1f} MiB downloaded", flush=True)
                                next_notice += 10 * 1024 * 1024
            with zipfile.ZipFile(temporary) as archive:
                bad = archive.testzip()
            if bad is not None:
                raise ValueError(f"Corrupt downloaded entry: {bad}")
            temporary.replace(path)
            print(f"Download complete: {written / 1024**2:.1f} MiB", flush=True)
            return
        except (requests.RequestException, zipfile.BadZipFile, ValueError):
            if attempt == 3:
                raise
            time.sleep(attempt)


@contextmanager
def open_dataset_archive(path):
    """Support both official wrapper ZIPs and the original UCI HAR Dataset.zip.

    Read members directly instead of extracting untrusted archive paths.
    """
    with zipfile.ZipFile(path) as outer:
        if any(name.endswith("/train/y_train.txt") for name in outer.namelist()):
            yield outer
        else:
            candidates = [name for name in outer.namelist()
                          if name.endswith("UCI HAR Dataset.zip")]
            if len(candidates) != 1:
                raise ValueError("Cannot locate the official UCI HAR Dataset.zip.")
            with zipfile.ZipFile(io.BytesIO(outer.read(candidates[0]))) as inner:
                yield inner


def read_member(archive, suffix, dtype):
    names = [name for name in archive.namelist() if name.endswith(suffix)]
    if len(names) != 1:
        raise ValueError(f"Expected exactly one archive entry ending in {suffix!r}")
    with archive.open(names[0]) as stream:
        return np.loadtxt(stream, dtype=dtype)


def read_official_split(archive, split):
    values = [read_member(archive, f"/{split}/Inertial Signals/{channel}_{split}.txt",
                          np.float32) for channel in CHANNELS]
    x = np.stack(values, axis=-1)
    y = read_member(archive, f"/{split}/y_{split}.txt", np.int64) - 1
    subjects = read_member(archive, f"/{split}/subject_{split}.txt", np.int64)
    if x.shape != (len(y), 128, 9) or subjects.shape != y.shape:
        raise ValueError(f"Unexpected source shape in {split}")
    if not np.isfinite(x).all() or sorted(np.unique(y).tolist()) != list(range(6)):
        raise ValueError(f"Invalid source signal or labels in {split}")
    return {"x": x, "y": y, "subjects": subjects}


def partition_subjects(subjects, seed=2026, validation_count=4):
    people = np.unique(subjects)
    if not 1 <= validation_count < len(people):
        raise ValueError("Validation person count must leave at least one training person.")
    shuffled = np.random.default_rng(seed).permutation(people)
    return sorted(shuffled[validation_count:].tolist()), sorted(shuffled[:validation_count].tolist())


def fit_standardizer(x):
    mean = x.mean(axis=(0, 1), dtype=np.float64)
    std = x.std(axis=(0, 1), dtype=np.float64)
    return mean, np.maximum(std, 1e-8)


def standardize(x, mean, std):
    return ((x - mean) / std).astype(np.float32)


def prepare(data_dir="data", split_seed=2026, validation_subjects=4, force=False):
    data_root = Path(data_dir)
    raw_archive = data_root / "raw" / "uci_har_official.zip"
    processed = data_root / "processed"
    download_archive(raw_archive)
    digest = sha256_file(raw_archive)
    if (processed / "metadata.json").is_file() and not force:
        saved = load_data(processed)["metadata"]
        if (saved["split_seed"] != split_seed or
                len(saved["split_subjects"]["val"]) != validation_subjects or
                saved["archive_sha256"] != digest):
            raise ValueError("Preparation settings/source changed. Use --force deliberately to regenerate.")
        print("Prepared data already match this archive and split. Test remains sealed.", flush=True)
        return saved
    with open_dataset_archive(raw_archive) as archive:
        print("Reading 9 preprocessed inertial channels (not the 561 engineered features)...", flush=True)
        development = read_official_split(archive, "train")
        held_out = read_official_split(archive, "test")
    if len(development["y"]) != 7352 or len(held_out["y"]) != 2947:
        raise ValueError("Official UCI HAR sample counts have changed; review before proceeding.")
    train_people, val_people = partition_subjects(development["subjects"], split_seed, validation_subjects)
    test_people = sorted(np.unique(held_out["subjects"]).tolist())
    if len(train_people) + len(val_people) != 21 or len(test_people) != 9:
        raise ValueError("Unexpected number of official subjects.")
    if set(train_people + val_people) & set(test_people):
        raise ValueError("Official train/test subjects overlap.")
    splits = {}
    for name, people in (("train", train_people), ("val", val_people)):
        mask = np.isin(development["subjects"], people)
        splits[name] = {key: value[mask] for key, value in development.items()}
        if sorted(np.unique(splits[name]["y"]).tolist()) != list(range(6)):
            raise ValueError(f"{name} does not contain all six activities.")
    mean, std = fit_standardizer(splits["train"]["x"])
    splits["test"] = held_out
    processed.mkdir(parents=True, exist_ok=True)
    for name, item in splits.items():
        item["x"] = standardize(item["x"], mean, std)
        destination = processed / f"{name}.npz"
        temporary = destination.with_suffix(".npz.part")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **item)
        temporary.replace(destination)
        print(f"{name}: {len(item['y'])} windows, {len(np.unique(item['subjects']))} people", flush=True)
    manifest = {
        "schema_version": 1,
        "source_url": SOURCE_URL,
        "source_page": SOURCE_PAGE,
        "dataset_doi": "10.24432/C54S4K",
        "dataset_license": "CC BY 4.0",
        "archive_sha256": digest,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split_seed": split_seed,
        "split_subjects": {"train": train_people, "val": val_people, "test": test_people},
        "sample_counts": {name: len(item["y"]) for name, item in splits.items()},
        "channel_order": CHANNELS,
        "labels": LABELS,
        "shape_per_sample": [128, 9],
        "sample_rate_hz": 50,
        "normalization": {"fit_split": "train", "axes": [0, 1], "ddof": 0,
                          "mean": mean.tolist(), "std": std.tolist()},
        "processed_sha256": {name: sha256_file(processed / f"{name}.npz") for name in splits},
        "test_policy": "Prepared with training statistics, excluded from default loading/training/model selection.",
    }
    temporary_manifest = processed / "metadata.json.part"
    temporary_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary_manifest.replace(processed / "metadata.json")
    load_data(processed)
    print("Preparation complete. No test predictions or test metrics were computed.", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--validation-subjects", type=int, default=4)
    parser.add_argument("--force", action="store_true", help="Explicitly regenerate processed arrays and split manifest.")
    args = parser.parse_args()
    prepare(args.data_dir, args.split_seed, args.validation_subjects, args.force)


if __name__ == "__main__":
    main()
