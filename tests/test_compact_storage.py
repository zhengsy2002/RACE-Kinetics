import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from race_kinetics import storage
from race_kinetics.evaluation import holdout
from race_kinetics.features.handcrafted import _load_or_build_features


class CompactStorageTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1] / 'data/cache/tests'
        root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=root)
        self.root = Path(self.temp.name)
        self.patch = patch.object(storage, 'DATA_ROOT', self.root / 'data')
        self.patch.start()
        storage.clear_runtime()

    def tearDown(self):
        storage.clear_runtime()
        self.patch.stop()
        self.temp.cleanup()

    def test_runtime_is_removed_after_exception_without_manifests(self):
        output = self.root / 'results/full'
        with self.assertRaisesRegex(RuntimeError, 'interrupted'):
            with storage.runtime_session():
                storage.freeze_record(output / 'identity.json', {'seed': 42})
                self.assertEqual(storage.read_record(output / 'identity.json'), {'seed': 42})
                raise RuntimeError('interrupted')
        self.assertFalse((self.root / 'data/runs').exists())
        self.assertIsNone(storage.read_record(output / 'identity.json'))

    def test_disk_records_are_not_loaded(self):
        output = self.root / 'results/full'
        path = output / 'identity.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({'seed': 42}))
        before = path.read_bytes()
        self.assertIsNone(storage.read_record(path))
        storage.write_record(output / 'new.json', {'value': 1})
        self.assertEqual(path.read_bytes(), before)

    def test_standalone_test_uses_model_metadata(self):
        n = 24
        frame = pd.DataFrame({'sample_id': [f's{i}' for i in range(n)], 'log10_value': np.sin(np.arange(n))})
        features = sparse.csr_matrix(np.random.default_rng(42).normal(size=(n, 3)))
        assignments = np.r_[np.arange(20) % 10, [-1] * 4]
        settings = dict(n_estimators=3, min_samples_leaf=2, max_features=0.5, n_jobs=1)
        output = self.root / 'results/full/seed_42'
        holdout.fit_final(frame, features, assignments, settings, 42, output, {'dataset': 'a'})
        storage.clear_runtime()
        holdout.test_final(frame, features, assignments, 42, output, 'random', {'dataset': 'a'})
        self.assertEqual(len(pd.read_csv(output / 'predictions/test.csv')), 4)
        storage.clear_runtime()
        with self.assertRaises(ValueError):
            holdout.fit_final(frame, features, assignments, settings, 42, output, {'dataset': 'b'})
        self.assertFalse((self.root / 'data/runs').exists())

    def test_plain_cache_preserves_values_and_rejects_drift(self):
        frame = pd.DataFrame({'reactant_smiles': ['CCO', 'CC'], 'product_smiles': ['CC=O', 'CCC'],
                              'ph': [7, None], 'temperature': [25, 30]})
        folder = storage.handcrafted_dir({})
        self.assertEqual(folder, self.root / 'data/features/handcrafted')
        args = dict(output_dir=folder, protocol={'dataset_sha256': 'a'}, morgan_bits=16, force_rebuild=False)
        first, *_ = _load_or_build_features(frame, **args)
        second, *_ = _load_or_build_features(frame, **args)
        self.assertEqual((first != second).nnz, 0)
        with self.assertRaisesRegex(RuntimeError, 'incompatible'):
            _load_or_build_features(frame, **{**args, 'protocol': {'dataset_sha256': 'b'}})
        self.assertFalse((folder / '.BUILDING.lock').exists())


if __name__ == '__main__':
    unittest.main()
