import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from race_kinetics.config import PROJECT_ROOT, load_config, validate


class ConfigTests(unittest.TestCase):
    def test_current_defaults_are_valid(self):
        config = load_config(PROJECT_ROOT / "configs/default.json")
        self.assertEqual(config["evaluation"]["n_estimators"], 200)
        self.assertEqual(config["loss"], {"enzyme_reaction": 1., "reactant_product": 1., "direction": 1.})

    def test_server_legacy_sections_are_identified(self):
        config = json.loads((PROJECT_ROOT / "configs/default.json").read_text())
        config.pop("downloads")
        config.pop("supervision")
        config.update(pretraining={}, study={})
        root = PROJECT_ROOT / "data/cache/tests"
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as temporary:
            path = Path(temporary) / "server.json"
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "unknown=\\['pretraining', 'study'\\]") as error:
                load_config(path)
            self.assertIn(str(path), str(error.exception))
            self.assertIn("missing=[]", str(error.exception))

    def test_missing_source_section_and_unknown_nested_option_are_identified(self):
        base = load_config(PROJECT_ROOT / "configs/default.json")
        missing = copy.deepcopy(base)
        missing.pop("data")
        with self.assertRaisesRegex(ValueError, "missing=\\['data'\\]"):
            validate(missing)
        base["evaluation"]["obsolete"] = True
        with self.assertRaisesRegex(ValueError, "Invalid evaluation: missing=\\[\\], unknown=\\['obsolete'\\]"):
            validate(base)


if __name__ == "__main__":
    unittest.main()
