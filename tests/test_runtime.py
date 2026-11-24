import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from race_kinetics.config import PROJECT_ROOT
from race_kinetics import runtime
from race_kinetics.storage import runtime_session


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        runtime.cleanup()
        root = PROJECT_ROOT / "data/cache/tests"
        root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=root)
        self.root = Path(self.temp.name)

    def tearDown(self):
        runtime.cleanup()
        self.temp.cleanup()

    def test_lazy_directories_and_environment_restoration(self):
        previous = os.environ.get("HF_HOME")
        with runtime_session():
            self.assertIsNone(runtime._SESSION)
            folder = runtime.configure()
            self.assertEqual(list(folder.iterdir()), [])
            self.assertTrue(Path(os.environ["HF_HOME"]).is_relative_to(folder))
            self.assertFalse(Path(os.environ["HF_HOME"]).exists())
            self.assertEqual(tempfile.tempdir, str(folder))
            self.assertEqual(runtime.configure(), folder)
        self.assertFalse(folder.exists())
        self.assertEqual(os.environ.get("HF_HOME"), previous)

    def test_exception_cleans_scratch_but_keeps_features_and_other_runs(self):
        persistent = self.root / "features.npy"
        persistent.write_bytes(b"necessary feature")
        with self.assertRaisesRegex(RuntimeError, "training failed"):
            with runtime_session():
                folder = runtime.configure()
                other = folder.parent / "run-other-test"
                other.mkdir()
                (folder / "temporary.txt").write_text("temporary")
                raise RuntimeError("training failed")
        self.assertFalse(folder.exists())
        self.assertTrue(other.is_dir())
        self.assertEqual(persistent.read_bytes(), b"necessary feature")
        other.rmdir()

    def test_atomic_files_stay_in_shared_scratch_and_failure_preserves_old_output(self):
        destination = self.root / "result.json"
        destination.write_text("old")
        with runtime_session():
            with self.assertRaisesRegex(ValueError, "failed write"):
                with runtime.atomic_output(destination) as temporary:
                    self.assertEqual(temporary.parent, runtime._SESSION)
                    temporary.write_text("incomplete")
                    raise ValueError("failed write")
            self.assertFalse(temporary.exists())
            self.assertEqual(destination.read_text(), "old")
            with runtime.atomic_output(destination) as temporary:
                temporary.write_text("complete")
        self.assertEqual(destination.read_text(), "complete")
        self.assertFalse(temporary.exists())

    def test_interrupt_and_sigterm_cleanup(self):
        for error in (KeyboardInterrupt(), SystemExit(143)):
            with self.subTest(error=type(error).__name__), self.assertRaises(type(error)):
                with runtime_session():
                    folder = runtime.configure()
                    raise error
            self.assertFalse(folder.exists())

    def test_forked_child_cannot_delete_parent_temporary_directory(self):
        folder = runtime.configure()
        owner = os.getpid()
        with patch.object(runtime.os, "getpid", return_value=owner + 1):
            runtime.cleanup()
        self.assertTrue(folder.exists())
        runtime.cleanup()
        self.assertFalse(folder.exists())


if __name__ == "__main__":
    unittest.main()
