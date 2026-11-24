import copy
import gc
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from race_kinetics.config import load_config, PROJECT_ROOT
from race_kinetics.data.caches import _sequence_sha1
from race_kinetics.features.matrix import build_matrix
from race_kinetics.features.alignment import _assert_pretraining_complete
from race_kinetics.pipeline import training_args
from race_kinetics.training.trainer import train
from race_kinetics.training import trainer
from race_kinetics.evaluation.holdout import run_evaluation
from race_kinetics import storage


class PipelineTest(unittest.TestCase):
    def test_pretraining_feature_extraction_and_heldout_regression(self):
        base = PROJECT_ROOT / "data/cache/tests"
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as temporary:
            root = Path(temporary)
            config = load_config(PROJECT_ROOT / "configs/default.json")
            config = copy.deepcopy(config)
            config["training"].update(device="cpu", epochs=1, batch_size=4, num_workers=0)
            config["evaluation"].update(n_estimators=4, n_jobs=1, batch_size=8)
            config["output_dir"] = str(root / "results/pretraining/full")
            data = config["data"]
            for key, folder in (("prepared_dir", "prepared"), ("protein_pretrain_cache", "pretrain_protein"),
                                ("protein_downstream_cache", "downstream_protein"), ("molformer_cache", "molecules")):
                data[key] = str(root / folder)
                Path(data[key]).mkdir()
            frame = pd.DataFrame({"sample_id": [f"id{i}" for i in range(24)],
                "sequence": ["M" + "A" * (i + 2) for i in range(24)],
                "reactant_smiles": ["C", "CC", "CCC"] * 8,
                "product_smiles": ["CO", "CCO", "CCCO"] * 8,
                "ph": [7.] * 24, "temperature": [25.] * 24,
                "log10_value": np.sin(np.arange(24))})
            for key in ("raw_csv", "downstream_csv"):
                data[key] = str(root / f"{key}.csv")
                frame.to_csv(data[key], index=False)
            prepared = Path(data["prepared_dir"])
            reactions = pd.DataFrame({"reaction_row": [0, 1, 2], "rhea_id": ["1", "2", "3"],
                "substrate": ["C", "CC", "CCC"], "product": ["CO", "CCO", "CCCO"]})
            reactions.to_csv(prepared / "reaction_index.tsv", sep="\t", index=False)
            pd.DataFrame({"protein_row": list(range(6)), "reaction_row": [0, 0, 1, 1, 2, 2],
                "rhea_id": ["1", "1", "2", "2", "3", "3"], "split": ["train"] * 4 + ["valid"] * 2
            }).to_csv(prepared / "pairs.tsv", sep="\t", index=False)
            rng = np.random.default_rng(42)
            protein_meta = {"complete": True, "input_key": "normalized_sequence_sha1", "hidden_size": 1152,
                "backbone": "ESMC-600M", "backend": "transformers", "model_dir": "synthetic-fixture",
                "pooling": "fixture", "model_dtype": "float32", "architecture": "fixture"}
            for key, count, mapping in (("protein_pretrain_cache", 6, "protein_row_to_cache_row.npy"),
                                        ("protein_downstream_cache", 24, "sample_row_to_cache_row.npy")):
                folder = Path(data[key])
                np.save(folder / "unique_embeddings.npy", rng.normal(size=(count, 1152)).astype(np.float32))
                np.save(folder / mapping, np.arange(count, dtype=np.int64))
                pd.DataFrame({"cache_row": np.arange(count), "sequence_sha1": frame.sequence.iloc[:count].map(_sequence_sha1)
                }).to_csv(folder / "unique_sequences.csv", index=False)
                (folder / "metadata.json").write_text(json.dumps(protein_meta))
            folder = Path(data["molformer_cache"])
            pd.DataFrame({"cache_row": range(6), "side_smiles": ["C", "CC", "CCC", "CO", "CCO", "CCCO"]
            }).to_csv(folder / "unique_sides.csv", index=False)
            np.save(folder / "side_embeddings.npy", rng.normal(size=(6, 768)).astype(np.float32))
            (folder / "metadata.json").write_text(json.dumps({"complete": True,
                "config": {"embedding_dim": 768, "model": "synthetic-fixture", "pooling": "fixture"}}))
            threads = torch.get_num_threads()
            torch.set_num_threads(1)
            try:
                with patch.object(storage, "DATA_ROOT", root / "data"), patch("race_kinetics.features.matrix.DATA_ROOT", root / "data"):
                    storage.write_record(Path(config["output_dir"]) / 'pretraining_binding.json', {'fixture': 'synthetic'})
                    train(training_args(config))
                    checkpoint = Path(config["output_dir"]) / "weights/best_reaction_alignment.pt"
                    storage.clear_runtime()
                    self.assertFalse((checkpoint.parent / 'last.pt').exists())
                    self.assertFalse((root / 'data/runs').exists())
                    self.assertTrue(_assert_pretraining_complete(checkpoint)["complete"])
                    matrix, path = build_matrix(config, frame, checkpoint)
                    self.assertTrue((root / 'data/features/handcrafted/handcrafted_features.npz').is_file())
                    self.assertEqual(matrix.shape, (24, 2823))
                    no_environment = copy.deepcopy(config)
                    no_environment["features"]["drop_assay"] = ["ph", "temperature"]
                    reduced, _ = build_matrix(no_environment, frame, checkpoint)
                    self.assertEqual(reduced.shape, (24, 2819))
                    learned_only = copy.deepcopy(config)
                    learned_only["features"]["mode"] = "learned_only"
                    learned, _ = build_matrix(learned_only, frame, checkpoint)
                    self.assertEqual(learned.shape, (24, 768))
                    np.testing.assert_array_equal(learned.toarray(), matrix[:, -768:].toarray())
                    assignments = np.array(list(range(10)) * 2 + [-1] * 4)
                    split = root / "split.csv"
                    pd.DataFrame({"sample_id": frame.sample_id, "inner_fold": assignments}).to_csv(split, index=False)
                    output = root / "results/random/full/seed_42"
                    run_evaluation(config, frame, matrix, assignments, split, path, "all", 42, "random", output)
                    result = json.loads((output / "evaluation/test.json").read_text())
                    self.assertEqual(result["test_rows"], 4)
                    self.assertEqual(len(pd.read_csv(output / "predictions/development_oof.csv")), 20)
                    checkpoint_time = (output / "weights/regressor.joblib").stat().st_mtime_ns
                    storage.clear_runtime()
                    run_evaluation(config, frame, matrix, assignments, split, path, "all", 42, "random", output)
                    self.assertEqual((output / "weights/regressor.joblib").stat().st_mtime_ns, checkpoint_time)
                    self.assertFalse((root / 'data/runs').exists())
                    self.assertFalse((root / 'data/audits').exists())
                    for name, key in [('no_er', 'enzyme_reaction'), ('no_rp', 'reactant_product'), ('no_dir', 'direction')]:
                        ablation = copy.deepcopy(config)
                        ablation['loss'][key] = 0.0
                        ablation['output_dir'] = str(root / 'results/pretraining' / name)
                        storage.write_record(Path(ablation['output_dir']) / 'pretraining_binding.json', {'fixture': name})
                        train(training_args(ablation))
                        saved = torch.load(Path(ablation['output_dir']) / 'weights/best_reaction_alignment.pt',
                                           weights_only=False)
                        self.assertEqual(saved['loss_weights'][key], 0.0)
                        self.assertEqual(saved['training_completion']['last_epoch'], 1)
                    outcomes = []
                    finish = trainer._finish_weights
                    def capture(args, settings, last):
                        outcomes.append({k: v.detach().clone() for k, v in last['full_model'].items()})
                        finish(args, settings, last)
                    for interrupted in (False, True):
                        repeat = copy.deepcopy(config)
                        repeat['training']['epochs'] = 2
                        repeat['output_dir'] = str(root / ('resumed' if interrupted else 'continuous'))
                        binding_path = Path(repeat['output_dir']) / 'pretraining_binding.json'
                        storage.write_record(binding_path, {'fixture': 'resume'})
                        args = training_args(repeat)
                        with patch.object(trainer, '_finish_weights', side_effect=capture):
                            if interrupted:
                                actual_epoch = trainer.run_epoch
                                calls = 0
                                def interrupt(*args, **kwargs):
                                    nonlocal calls
                                    calls += 1
                                    if calls == 3:
                                        raise RuntimeError('simulated interruption')
                                    return actual_epoch(*args, **kwargs)
                                with patch.object(trainer, 'run_epoch', side_effect=interrupt):
                                    with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                                        train(args)
                                storage.clear_runtime()
                                storage.write_record(binding_path, {'fixture': 'resume'})
                            train(args)
                    for key in outcomes[0]:
                        torch.testing.assert_close(outcomes[0][key], outcomes[1][key], rtol=0, atol=0)
            finally:
                torch.set_num_threads(threads)
                gc.collect()
