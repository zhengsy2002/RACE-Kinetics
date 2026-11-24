import atexit
from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import signal
import tempfile
import threading
import warnings

from .config import project_path

_SESSION = None
_ENVIRONMENT = {}
_TEMP_DIRECTORY = None
_OWNER_PID = None


def configure():
    global _SESSION, _TEMP_DIRECTORY, _OWNER_PID
    if _SESSION is not None:
        return _SESSION
    root = project_path("data/tmp")
    root.mkdir(parents=True, exist_ok=True)
    _SESSION = Path(tempfile.mkdtemp(prefix=f"run-{os.getpid()}-", dir=root)).resolve()
    _OWNER_PID = os.getpid()
    paths = {
        "HF_HOME": "huggingface", "HF_HUB_CACHE": "huggingface/hub",
        "HUGGINGFACE_HUB_CACHE": "huggingface/hub", "HF_MODULES_CACHE": "huggingface/modules",
        "HF_XET_CACHE": "huggingface/xet", "HF_TOKEN_PATH": "huggingface/token",
        "TRANSFORMERS_CACHE": "huggingface/hub", "TORCH_HOME": "torch",
        "TORCH_EXTENSIONS_DIR": "torch/extensions",
        "TMPDIR": "", "TEMP": "", "TMP": "", "JOBLIB_TEMP_FOLDER": "",
        "XDG_CACHE_HOME": "runtime", "CUDA_CACHE_PATH": "runtime/cuda",
        "TRITON_CACHE_DIR": "runtime/triton", "TORCHINDUCTOR_CACHE_DIR": "runtime/inductor",
    }
    values = {key: str(_SESSION / relative) for key, relative in paths.items()}
    values.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    _ENVIRONMENT.update({key: os.environ.get(key) for key in values})
    os.environ.update(values)
    _TEMP_DIRECTORY = tempfile.tempdir
    tempfile.tempdir = str(_SESSION)
    return _SESSION


def cleanup():
    global _SESSION, _TEMP_DIRECTORY, _OWNER_PID
    if _SESSION is None or _OWNER_PID != os.getpid():
        return
    folder = _SESSION
    for key, value in _ENVIRONMENT.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    _ENVIRONMENT.clear()
    tempfile.tempdir = _TEMP_DIRECTORY
    _SESSION = None
    _TEMP_DIRECTORY = None
    _OWNER_PID = None
    root = project_path("data/tmp")
    if folder.resolve().parent != root or not folder.name.startswith(f"run-{os.getpid()}-"):
        raise ValueError(f"Refusing to clean an unowned temporary directory: {folder}")
    try:
        shutil.rmtree(folder)
    except FileNotFoundError:
        pass
    except OSError as error:
        warnings.warn(f"Temporary files could not be fully removed from {folder}: {error}")
    try:
        root.rmdir()
    except OSError:
        pass


@contextmanager
def atomic_output(destination):
    destination = project_path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix="write-", suffix=destination.suffix, dir=configure())
    os.close(descriptor)
    temporary = Path(name)
    try:
        yield temporary
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def lifecycle():
    previous = None
    install = threading.current_thread() is threading.main_thread()
    if install:
        previous = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, _terminate)
    try:
        yield
    finally:
        cleanup()
        if install:
            signal.signal(signal.SIGTERM, previous)


def _terminate(signum, frame):
    raise SystemExit(128 + signum)


atexit.register(cleanup)
