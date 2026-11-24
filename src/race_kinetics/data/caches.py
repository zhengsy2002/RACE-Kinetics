from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
PROTEIN_DIM = 1152
PROTEIN_BACKBONE = "ESMC-600M/transformers"
def _normalize_sequence(value: object) -> str:
    compact = re.sub(r"\s+", "", str(value)).upper()
    return re.sub(r"[UZOB]", "X", compact)


def _sequence_sha1(value: object) -> str:
    return hashlib.sha1(_normalize_sequence(value).encode("utf-8")).hexdigest()


def _validate_sequence_cache(
    frame: pd.DataFrame, cache_dir: Path
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, object]]:
    required = [
        cache_dir / "unique_sequences.csv",
        cache_dir / "unique_embeddings.npy",
        cache_dir / "sample_row_to_cache_row.npy",
        cache_dir / "metadata.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Sequence-keyed protein cache is missing: {missing}")
    metadata = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
    if not metadata.get("complete", False):
        raise RuntimeError(f"Sequence-keyed protein cache is incomplete: {cache_dir}")
    if metadata.get("input_key") != "normalized_sequence_sha1":
        raise RuntimeError(
            "Expected sequence-SHA1 cache semantics, got "
            f"{metadata.get('input_key')!r}"
        )
    manifest = pd.read_csv(cache_dir / "unique_sequences.csv", dtype=str).fillna("")
    if not {"cache_row", "sequence_sha1"}.issubset(manifest.columns):
        raise KeyError("unique_sequences.csv requires cache_row and sequence_sha1")
    manifest["cache_row"] = pd.to_numeric(manifest["cache_row"], errors="raise").astype(int)
    if manifest["cache_row"].tolist() != list(range(len(manifest))):
        raise RuntimeError("Sequence cache rows must be contiguous and ordered")
    embeddings = np.load(cache_dir / "unique_embeddings.npy", mmap_mode="r")
    row_mapping = np.load(cache_dir / "sample_row_to_cache_row.npy", mmap_mode="r")
    expected_dim = int(metadata.get("hidden_size", 0))
    if embeddings.ndim != 2 or embeddings.shape[0] != len(manifest):
        raise RuntimeError(f"Invalid sequence embedding matrix: {embeddings.shape}")
    if expected_dim <= 0 or embeddings.shape[1] != expected_dim:
        raise RuntimeError(
            f"Sequence embedding dimension mismatch: {embeddings.shape[1]} != {expected_dim}"
        )
    if row_mapping.shape != (len(frame),):
        raise RuntimeError(
            f"kcat sequence row mapping mismatch: {row_mapping.shape} != {(len(frame),)}"
        )
    if row_mapping.dtype.kind not in "iu" or not manifest["sequence_sha1"].is_unique:
        raise ValueError("Expected integer row mapping and unique sequence hashes")
    row_mapping_array = np.asarray(row_mapping, dtype=np.int64)
    if row_mapping_array.min() < 0 or row_mapping_array.max() >= len(manifest):
        raise RuntimeError("kcat sequence row mapping is out of range")
    expected_hashes = frame["sequence"].map(_sequence_sha1).to_numpy(str)
    actual_hashes = manifest["sequence_sha1"].to_numpy(str)[row_mapping_array]
    mismatch = np.flatnonzero(expected_hashes != actual_hashes)
    if len(mismatch):
        raise RuntimeError(
            "kcat sequence cache does not match the current CSV; first mismatched "
            f"rows={mismatch[:5].tolist()}"
        )
    return embeddings, row_mapping_array, manifest, metadata
def validate_protein_metadata(metadata: dict) -> None:
    if not metadata.get("complete") or metadata.get("input_key") != "normalized_sequence_sha1":
        raise ValueError("Incomplete or invalid sequence-keyed protein cache")
    if metadata.get("hidden_size") != PROTEIN_DIM:
        raise ValueError(f"Expected {PROTEIN_BACKBONE} hidden_size={PROTEIN_DIM}; got {metadata.get('hidden_size')}")
    if metadata.get("backbone") != "ESMC-600M" or metadata.get("backend") != "transformers":
        raise ValueError(f"Expected ESMC-600M/transformers cache; got {metadata.get('backbone')}/{metadata.get('backend')}")
    for key in ("model_dir", "pooling", "model_dtype", "architecture"):
        if not metadata.get(key):
            raise ValueError(f"ESM-C metadata is missing {key}")
def load_side_cache(cache_dir: Path) -> tuple[np.ndarray, dict[str, int], dict[str, object]]:
    required = [
        cache_dir / "unique_sides.csv",
        cache_dir / "side_embeddings.npy",
        cache_dir / "metadata.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"MolFormer side cache is missing: {missing}")
    metadata = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
    if not metadata.get(
        "complete", False
    ):
        raise RuntimeError(f"Incomplete or unsupported MolFormer cache: {cache_dir}")
    manifest = pd.read_csv(cache_dir / "unique_sides.csv", keep_default_na=False)
    if not {"cache_row", "side_smiles"}.issubset(manifest.columns):
        raise KeyError("MolFormer unique_sides.csv requires cache_row and side_smiles")
    rows = pd.to_numeric(manifest["cache_row"], errors="raise").astype(int).to_numpy()
    if not np.array_equal(rows, np.arange(len(manifest))):
        raise RuntimeError("MolFormer side cache rows are not contiguous")
    embeddings = np.load(cache_dir / "side_embeddings.npy", mmap_mode="r")
    expected_dim = int(metadata["config"]["embedding_dim"])
    if expected_dim != 768 or not manifest["side_smiles"].is_unique:
        raise ValueError("Expected unique molecule sides and 768-dimensional MolFormer embeddings")
    if embeddings.shape != (len(manifest), expected_dim):
        raise RuntimeError(f"Invalid MolFormer side embeddings: {embeddings.shape}")
    lookup = dict(zip(manifest["side_smiles"].astype(str), rows.tolist()))
    return embeddings, lookup, metadata
