"""Small correctness checks for the comparison protocol; no test-set scoring."""

from pathlib import Path
from copy import deepcopy
import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from har_compare.metrics import classification_metrics
from har_compare.models import build_model
from har_compare.runtime import metadata_fingerprint


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"


class MetricsTests(unittest.TestCase):
    def test_hand_calculated_confusion_and_macro_f1(self):
        result = classification_metrics(
            np.array([0, 0, 1, 1, 2, 2]),
            np.array([0, 1, 1, 1, 0, 2]),
            num_classes=3,
        )
        np.testing.assert_array_equal(
            result["confusion_matrix"], [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
        )
        self.assertAlmostEqual(result["accuracy"], 4 / 6)
        self.assertAlmostEqual(result["macro_f1"], (0.5 + 0.8 + 2 / 3) / 3)

    def test_unpredicted_classes_remain_in_macro_average(self):
        result = classification_metrics(
            np.array([0, 1, 2, 3, 4, 5]), np.zeros(6, dtype=np.int64)
        )
        self.assertAlmostEqual(result["accuracy"], 1 / 6)
        self.assertAlmostEqual(result["macro_f1"], (2 / 7) / 6)
        self.assertTrue(np.isfinite(result["macro_f1"]))


class ModelTests(unittest.TestCase):
    def test_all_models_produce_logits_and_finite_gradients(self):
        torch.set_num_threads(2)
        for name in ("cnn", "rnn", "lstm"):
            with self.subTest(model=name):
                torch.manual_seed(42)
                model = build_model(name)
                x = torch.randn(4, 128, 9)
                target = torch.tensor([0, 1, 2, 5])
                logits = model(x)
                self.assertEqual(tuple(logits.shape), (4, 6))
                self.assertTrue(torch.isfinite(logits).all())
                loss = torch.nn.functional.cross_entropy(logits, target)
                loss.backward()
                gradients = [p.grad for p in model.parameters() if p.requires_grad]
                self.assertTrue(all(g is not None for g in gradients))
                self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0)


class DataFingerprintTests(unittest.TestCase):
    def setUp(self):
        self.metadata = {
            "schema_version": 1,
            "archive_sha256": "a" * 64,
            "split_seed": 2026,
            "split_subjects": {"train": [1, 2], "val": [3], "test": [4]},
            "sample_counts": {"train": 4, "val": 2, "test": 2},
            "channel_order": [f"channel_{i}" for i in range(9)],
            "labels": [f"class_{i}" for i in range(6)],
            "shape_per_sample": [128, 9],
            "sample_rate_hz": 50,
            "normalization": {
                "fit_split": "train", "axes": [0, 1], "ddof": 0,
                "mean": [0.0] * 9, "std": [1.0] * 9,
            },
            "created_at_utc": "2026-11-09T00:00:00+00:00",
            "processed_sha256": {"train": "b" * 64},
        }

    def test_repackaging_and_creation_time_do_not_change_identity(self):
        repacked = deepcopy(self.metadata)
        repacked["created_at_utc"] = "2026-11-10T00:00:00+00:00"
        repacked["processed_sha256"] = {"train": "c" * 64}
        # Mapping insertion order is not part of a dataset's meaning either.
        repacked = dict(reversed(list(repacked.items())))
        self.assertEqual(metadata_fingerprint(self.metadata), metadata_fingerprint(repacked))

    def test_actual_input_changes_invalidate_identity(self):
        changed_normalization = deepcopy(self.metadata)
        changed_normalization["normalization"]["mean"][0] = 0.1
        changed_people = deepcopy(self.metadata)
        changed_people["split_subjects"]["train"] = [1, 3]
        changed_people["split_subjects"]["val"] = [2]
        changed_channels = deepcopy(self.metadata)
        changed_channels["channel_order"].reverse()
        changed_archive = deepcopy(self.metadata)
        changed_archive["archive_sha256"] = "d" * 64
        reference = metadata_fingerprint(self.metadata)
        for name, changed in (
            ("normalization", changed_normalization),
            ("subject_split", changed_people),
            ("channel_order", changed_channels),
            ("source_archive", changed_archive),
        ):
            with self.subTest(change=name):
                self.assertNotEqual(reference, metadata_fingerprint(changed))

    def test_missing_provenance_cannot_silently_match(self):
        incomplete = deepcopy(self.metadata)
        del incomplete["normalization"]
        with self.assertRaises(ValueError):
            metadata_fingerprint(incomplete)


@unittest.skipUnless(
    (PROCESSED / "metadata.json").exists(),
    "Run python -m har_compare.prepare --data-dir data before data checks.",
)
class DataProtocolTests(unittest.TestCase):
    def test_cache_manifest_checksum_mismatch_is_rejected(self):
        from har_compare.data import load_data

        metadata = json.loads((PROCESSED / "metadata.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory(prefix="har_cache_check_") as directory:
            temporary = Path(directory)
            for split in ("train", "val"):
                source = PROCESSED / f"{split}.npz"
                destination = temporary / source.name
                try:
                    os.link(source, destination)
                except OSError:
                    shutil.copyfile(source, destination)
            manifest = temporary / "metadata.json"
            manifest.write_text(json.dumps(metadata), encoding="utf-8")
            # No test.npz is present: normal training must still load successfully.
            self.assertNotIn("test", load_data(temporary))
            # Change only the temporary manifest, never the linked real arrays.
            metadata["processed_sha256"]["train"] = "0" * 64
            manifest.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                load_data(temporary)

    def test_default_loading_never_opens_test_arrays(self):
        from har_compare.data import load_data

        opened = []
        original_load = np.load

        def recording_load(file, *args, **kwargs):
            if isinstance(file, (str, Path)):
                opened.append(Path(file).name.lower())
            return original_load(file, *args, **kwargs)

        with patch("numpy.load", side_effect=recording_load):
            data = load_data(PROCESSED)
        self.assertNotIn("test", data)
        self.assertFalse(any("test" in filename for filename in opened), opened)

    def test_subjects_are_disjoint_and_training_is_standardized(self):
        from har_compare.data import load_data

        # Opening test arrays here checks identities and dimensions, not predictions.
        data = load_data(PROCESSED, include_test=True)
        subject_sets = {}
        for split in ("train", "val", "test"):
            part = data[split]
            subject_sets[split] = set(np.asarray(part["subjects"]).tolist())
            self.assertEqual(part["x"].shape[1:], (128, 9))
            self.assertEqual(len(part["x"]), len(part["y"]))
            self.assertEqual(len(part["x"]), len(part["subjects"]))
            self.assertEqual(part["x"].dtype, np.float32)
            self.assertTrue(np.isfinite(part["x"]).all())
            self.assertEqual(set(np.asarray(part["y"]).tolist()), set(range(6)))
        self.assertEqual(len(subject_sets["train"]), 17)
        self.assertEqual(len(subject_sets["val"]), 4)
        self.assertEqual(len(subject_sets["test"]), 9)
        self.assertTrue(subject_sets["train"].isdisjoint(subject_sets["val"]))
        self.assertTrue(subject_sets["train"].isdisjoint(subject_sets["test"]))
        self.assertTrue(subject_sets["val"].isdisjoint(subject_sets["test"]))
        self.assertEqual(len(data["train"]["x"]) + len(data["val"]["x"]), 7352)
        self.assertEqual(len(data["test"]["x"]), 2947)
        train = data["train"]["x"].astype(np.float64)
        np.testing.assert_allclose(train.mean(axis=(0, 1)), 0, atol=1e-4)
        np.testing.assert_allclose(train.std(axis=(0, 1)), 1, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
