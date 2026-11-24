from __future__ import annotations
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    from .runtime import atomic_output
    with atomic_output(path) as temp:
        temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def freeze_json(path: Path, value: object) -> None:
    value = json.loads(json.dumps(value))
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != value:
        raise RuntimeError(f"Frozen inputs/configuration changed: {path}. Use a new suite directory.")
    if not path.exists():
        write_json(path, value)
def capture_rng_state(generator: torch.Generator) -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "loader": generator.get_state(),
    }


def restore_rng_state(state: dict[str, object], generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    generator.set_state(state["loader"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def atomic_torch_save(value: object, path: Path) -> None:
    from .runtime import atomic_output
    with atomic_output(path) as temporary:
        torch.save(value, temporary)


def _mean_metrics(rows: list[dict[str, float]], prefix: str) -> dict[str, float]:
    if not rows:
        raise RuntimeError("No batches were processed")
    return {
        f"{prefix}_{key}": float(np.mean([row[key] for row in rows]))
        for key in rows[0]
    }
def move_batch(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }
def _signature(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def file_identity(files, config):
    from importlib.metadata import version
    import sys
    files = list(dict.fromkeys(Path(p).resolve() for p in files))
    package = Path(__file__).resolve().parent
    identity = {"config": config, "inputs": {str(p): sha256(p) for p in files},
                "runtime": {"python": sys.version, **{name: version(name) for name in
                            ("torch", "numpy", "pandas", "scipy", "scikit-learn", "rdkit")}},
                "code": {str(p.relative_to(package)): sha256(p) for p in sorted(package.rglob("*.py"))}}
    return identity


def bind_files(path, files, config):
    identity = file_identity(files, config)
    freeze_json(Path(path), identity)
    return identity


def bind_preparation_sources(path, files, config):
    package = Path(__file__).resolve().parent
    relevant = ("data/preparation.py", "data/kcat.py", "data/validation.py",
                "data/caches.py", "data/protocol.py", "features/handcrafted.py")
    inputs = {str(Path(p).resolve()): sha256(Path(p)) for p in files}
    code = {name: sha256(package / name) for name in relevant}
    path = Path(path)
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        old_code = {k.replace("\\", "/"): v for k, v in old.get("code", {}).items()}
        if old.get("inputs") != inputs or old.get("config") != config:
            raise RuntimeError("Prepared source inputs changed; use a new prepared directory")
        if any(old_code.get(k) != v for k, v in code.items()):
            raise RuntimeError("Preparation implementation changed; use a new prepared directory")
    else:
        write_json(path, {"inputs": inputs, "config": config, "code": code})
    return {"inputs": inputs, "config": config, "code": code}
