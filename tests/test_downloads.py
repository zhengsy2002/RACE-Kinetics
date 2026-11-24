import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
from safetensors.numpy import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from race_kinetics.config import PROJECT_ROOT, load_config, project_path, validate_paths
from race_kinetics.backbones import download
from race_kinetics.pipeline import extract_features, model_files
from race_kinetics import runtime


def encoder_fixture(folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    config = {"model_type": name, "hidden_size": 1152 if name == "esmc" else 768}
    if name == "molformer":
        config["auto_map"] = {"AutoModel": "modeling_molformer.MolformerModel"}
        (folder / "modeling_molformer.py").write_text("class MolformerModel: pass\n")
    for filename, value in [("config.json", config), ("tokenizer_config.json", {}), ("tokenizer.json", {})]:
        (folder / filename).write_text(json.dumps(value), encoding="utf-8")
    save_file({"fixture": np.zeros((2, 2), dtype=np.float32)}, str(folder / "model.safetensors"))


class DownloadTests(unittest.TestCase):
    def setUp(self):
        root = PROJECT_ROOT / "data/cache/tests"
        root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=root)
        self.root = Path(self.temp.name)
        self.config = load_config(PROJECT_ROOT / "configs/default.json")
        for name in ("esmc", "molformer"):
            self.config["data"][f"{name}_model"] = str(self.root / name)

    def tearDown(self):
        runtime.cleanup()
        self.temp.cleanup()

    def test_missing_models_download_then_reuse_without_network(self):
        def acquire(repo, revision, folder):
            self.assertEqual(len(revision), 40)
            self.assertTrue(folder.is_relative_to(PROJECT_ROOT))
            encoder_fixture(folder, folder.name)

        with patch.object(download, "configure_runtime"), patch.object(download, "_download", side_effect=acquire) as call:
            download.ensure_models(self.config)
            self.assertEqual(call.call_count, 2)
            download.ensure_models(self.config)
            self.assertEqual(call.call_count, 2)

    def test_missing_tokenizer_and_custom_code_are_incomplete(self):
        folder = Path(self.config["data"]["molformer_model"])
        encoder_fixture(folder, "molformer")
        (folder / "tokenizer.json").unlink()
        (folder / "modeling_molformer.py").unlink()
        self.assertEqual(download.model_status(folder, "molformer"), ["modeling_molformer.py", "tokenizer.json"])

    def test_truncated_weights_are_repaired_and_revalidated(self):
        folder = Path(self.config["data"]["esmc_model"])
        encoder_fixture(folder, "esmc")
        (folder / "model.safetensors").write_bytes(b"partial")
        with patch.object(download, "configure_runtime"), patch.object(download, "_download") as call:
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                download.ensure_models(self.config, ("esmc",))
            self.assertEqual(call.call_args.args[-1], folder)

    def test_disabled_download_fails_without_network(self):
        self.config["downloads"]["auto_download"] = False
        with patch.object(download, "configure_runtime"), patch.object(download, "_download") as call:
            with self.assertRaises(FileNotFoundError):
                download.ensure_models(self.config, ("esmc",))
            call.assert_not_called()

    def test_wrong_model_dimension_is_not_overwritten(self):
        folder = Path(self.config["data"]["esmc_model"])
        encoder_fixture(folder, "esmc")
        (folder / "config.json").write_text('{"model_type":"esmc","hidden_size":960}')
        with patch.object(download, "configure_runtime"), patch.object(download, "_download") as call:
            with self.assertRaisesRegex(ValueError, "Wrong esmc"):
                download.ensure_models(self.config, ("esmc",))
            call.assert_not_called()

    def test_subprocess_download_is_online_without_changing_parent(self):
        folder = self.root / "esmc"
        folder.mkdir()
        (folder / "tokenizer.json").write_text("invalid")
        with patch.dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1"), patch.object(subprocess, "run") as call:
            download._download("owner/model", "a" * 40, folder)
            options = call.call_args.kwargs
            self.assertEqual(options["env"]["HF_HUB_OFFLINE"], "0")
            self.assertEqual(options["env"]["TRANSFORMERS_OFFLINE"], "0")
            self.assertEqual(options["env"]["PYTHONPATH"], str(PROJECT_ROOT / "src"))
            self.assertIn("--scratch", call.call_args.args[0])
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")

    def test_network_failure_is_not_silently_ignored(self):
        with patch.object(subprocess, "run", side_effect=subprocess.CalledProcessError(1, ["download"])):
            with self.assertRaisesRegex(RuntimeError, "No encoder from another project"):
                download._download("owner/model", "a" * 40, self.root / "esmc")

    def test_external_sources_caches_and_splits_are_rejected(self):
        outside = PROJECT_ROOT.parent / "other-project/data"
        for key in self.config["data"]:
            config = copy.deepcopy(self.config)
            config["data"][key] = str(outside)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "inside this project"):
                validate_paths(config)
        config = copy.deepcopy(self.config)
        config["holdout"]["random"] = str(outside / "split.csv")
        with self.assertRaises(ValueError):
            validate_paths(config)
        with self.assertRaises(ValueError):
            project_path("../other-project/data")

    def test_sharded_weights_and_download_cache_exclusion(self):
        folder = Path(self.config["data"]["esmc_model"])
        encoder_fixture(folder, "esmc")
        (folder / "model.safetensors").rename(folder / "part-1.safetensors")
        (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"fixture": "part-1.safetensors"}}))
        self.assertEqual(download.model_status(folder, "esmc"), [])
        (folder / ".cache").mkdir()
        (folder / ".cache/download.json").write_text("{}")
        self.assertFalse(any(".cache" in p.parts for p in model_files(folder)))

    def test_pipeline_requests_encoder_when_features_are_missing(self):
        self.config["data"]["protein_pretrain_cache"] = str(self.root / "protein")
        with patch.object(download, "ensure_models", side_effect=RuntimeError("download sentinel")) as call:
            with self.assertRaisesRegex(RuntimeError, "download sentinel"):
                extract_features(self.config)
            call.assert_called_once_with(self.config, ("esmc",))

    def test_complete_features_do_not_request_encoders(self):
        for key in ("protein_pretrain_cache", "protein_downstream_cache", "molformer_cache"):
            folder = self.root / key
            folder.mkdir()
            (folder / "metadata.json").write_text('{"complete":true}')
            self.config["data"][key] = str(folder)
        with patch.object(download, "ensure_models") as call:
            extract_features(self.config)
            call.assert_not_called()

    def test_download_promotes_model_files_but_not_runtime_cache(self):
        scratch = runtime.configure() / "download-fixture"
        scratch.mkdir()
        folder = Path(self.config["data"]["esmc_model"])

        def snapshot(**kwargs):
            self.assertEqual(kwargs["local_dir"], scratch)
            encoder_fixture(scratch, "esmc")
            (scratch / ".cache").mkdir()
            (scratch / ".cache/metadata.json").write_text("{}")

        arguments = ["worker", "--repo-id", "owner/model", "--revision", "a" * 40,
                     "--folder", str(folder), "--scratch", str(scratch)]
        with patch.dict(sys.modules, huggingface_hub=SimpleNamespace(snapshot_download=snapshot)), patch.object(sys, "argv", arguments):
            download.main()
        self.assertEqual(download.model_status(folder, "esmc"), [])
        self.assertFalse((folder / ".cache").exists())
        runtime.cleanup()
        self.assertFalse(scratch.exists())
        self.assertTrue((folder / "model.safetensors").is_file())


if __name__ == "__main__":
    unittest.main()
