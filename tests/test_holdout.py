import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from race_kinetics.data.holdout import read_partition
from race_kinetics.evaluation import holdout
from race_kinetics import storage


class HoldoutTests(unittest.TestCase):
    def setUp(self):
        test_root = Path(__file__).resolve().parents[1] / "data/cache/tests"
        test_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=test_root)
        self.root = Path(self.temp.name)
        self.patch = patch.object(storage, "DATA_ROOT", self.root / "data")
        self.patch.start()
        n = 24
        self.frame = pd.DataFrame({"sample_id": [f"id{i}" for i in range(n)],
            "sequence": [f"SEQ{i}" for i in range(n)], "log10_value": np.sin(np.arange(n))})
        self.split = pd.DataFrame({"sample_id": self.frame.sample_id, "row_index": np.arange(n),
            "partition": ["development"] * 20 + ["test"] * 4,
            "inner_fold": [str(i % 10) for i in range(20)] + [""] * 4,
            "observation_group": [f"o{i}" for i in range(n)],
            "component": [f"c{i}" for i in range(n)],
            "sequence_sha256": [hashlib.sha256(s.encode()).hexdigest() for s in self.frame.sequence]})
        self.path = self.root / "split.csv"
        self.features = sparse.csr_matrix(np.random.default_rng(42).normal(size=(n, 12)).astype(np.float32))
        self.settings = {"n_estimators": 4, "max_features": 0.5, "min_samples_leaf": 2, "n_jobs": 1}
        self.assignments = np.array([i % 10 for i in range(20)] + [-1] * 4)

    def tearDown(self):
        self.patch.stop()
        self.temp.cleanup()

    def read(self, protocol="homology"):
        self.split.to_csv(self.path, index=False)
        return read_partition(self.frame, self.path, protocol)

    def test_supplied_assignments_and_reordering(self):
        self.split = self.split.sample(frac=1, random_state=7)
        _, assignments, report = self.read()
        np.testing.assert_array_equal(assignments, self.assignments)
        self.assertEqual(report["test_rows"], 4)

    def test_component_cannot_cross_holdout(self):
        self.split.loc[20, "component"] = "c0"
        with self.assertRaisesRegex(ValueError, "component crosses"):
            self.read()

    def test_component_cannot_cross_inner_folds(self):
        self.split.loc[1, "component"] = "c0"
        with self.assertRaisesRegex(ValueError, "component crosses inner"):
            self.read()

    def test_test_fold_and_row_index_rejected(self):
        self.split.loc[20, "inner_fold"] = "0"
        with self.assertRaisesRegex(ValueError, "Test records"):
            self.read()
        self.split.loc[20, "inner_fold"] = ""
        self.split.loc[0, "row_index"] = 1
        with self.assertRaisesRegex(ValueError, "row_index"):
            self.read()

    def test_sequence_drift_rejected(self):
        self.frame.loc[0, "sequence"] = "CHANGED"
        with self.assertRaisesRegex(ValueError, "hashes"):
            self.read()

    def test_duplicate_observation_rejected_in_random(self):
        self.split.loc[20, "observation_group"] = "o0"
        with self.assertRaisesRegex(ValueError, "observation_group crosses"):
            self.read("random")

    def test_end_to_end_and_test_labels_do_not_affect_fit(self):
        output = self.root / "results/random/full/seed_42"
        identity = {"test": "fixture"}
        holdout.cross_validate(self.frame, self.features, self.assignments, self.settings, 42, output, "random")
        oof = pd.read_csv(output / "predictions/development_oof.csv")
        self.assertEqual(set(oof.sample_id), set(self.frame.sample_id[:20]))
        metrics = pd.read_csv(output / "evaluation/inner_fold_metrics.csv")
        self.assertTrue((metrics.train_rows == 18).all())
        self.assertFalse((output / "evaluation/test.json").exists())
        holdout.fit_final(self.frame, self.features, self.assignments, self.settings, 42, output, identity)
        changed = self.frame.copy()
        changed.loc[20:, "log10_value"] = 10000
        other = self.root / "other"
        holdout.fit_final(changed, self.features, self.assignments, self.settings, 42, other, identity)
        model = joblib.load(output / "weights/regressor.joblib")
        model2 = joblib.load(other / "weights/regressor.joblib")
        np.testing.assert_array_equal(model.predict(self.features), model2.predict(self.features))
        holdout.test_final(self.frame, self.features, self.assignments, 42, output, "random", identity)
        test = pd.read_csv(output / "predictions/test.csv")
        self.assertEqual(set(test.sample_id), set(self.frame.sample_id[20:]))
        self.assertTrue(holdout.completed(output, "test"))
        with (output / "weights/regressor.joblib").open("ab") as f:
            f.write(b"tampered")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            holdout.test_final(self.frame, self.features, self.assignments, 42, output, "random", identity)

    def test_test_requires_training_receipt(self):
        with self.assertRaises(FileNotFoundError):
            holdout.test_final(self.frame, self.features, self.assignments, 42, self.root / "empty", "random", {})


if __name__ == "__main__":
    unittest.main()
