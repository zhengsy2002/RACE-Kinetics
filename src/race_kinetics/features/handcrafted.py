from pathlib import Path
from typing import Any
from importlib.metadata import version
import json
import numpy as np
import pandas as pd
from scipy import sparse
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from ..utils import write_json as _write_json, sha256 as _sha256

def _numeric_with_missing(frame: pd.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray]:
    if column not in frame.columns:
        values = np.zeros(len(frame), dtype=np.float32)
        missing = np.ones(len(frame), dtype=np.float32)
        return values, missing
    numeric = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float32)
    missing = (~np.isfinite(numeric)).astype(np.float32)
    numeric[~np.isfinite(numeric)] = 0.0
    return numeric, missing


def _validate_protocol(frame, data_path):
    required = {"sample_id", "sequence", "log10_value", "reactant_smiles", "product_smiles"}
    if not required.issubset(frame):
        raise ValueError(f"Missing native kcat fields: {required - set(frame)}")
    if frame.sample_id.isna().any() or not frame.sample_id.is_unique or not np.isfinite(frame.log10_value).all():
        raise ValueError("Invalid sample IDs or targets")
    return {"dataset_sha256": _sha256(data_path), "rows": len(frame), "target": "log10_value",
            "partition_source": "external split CSV"}

def _build_feature_matrix(frame, *, morgan_bits=1024, use_product_missing=False):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=morgan_bits)
    left = frame.reactant_smiles.fillna("").astype(str).tolist()
    right = frame.product_smiles.fillna("").astype(str).tolist()
    right = [b if b.strip() else a for a,b in zip(left,right)]
    cache = {}
    for text in dict.fromkeys(left + right):
        vector = np.zeros(morgan_bits, dtype=np.float32)
        try:
            mol = Chem.MolFromSmiles(text) if text.strip() and text.lower() != "nan" else None
        except Exception:
            mol = None
        if mol is not None:
            DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), vector)
        cache[text] = (vector, float(mol is not None))
    ph, pm = _numeric_with_missing(frame, "ph")
    temp, tm = _numeric_with_missing(frame, "temperature")
    product_missing = np.zeros(len(frame), dtype=np.float32)
    if use_product_missing:
        if "product_missing" not in frame:
            raise ValueError("Original product_missing is required; recover it from raw_csv before fallback")
        product_missing = pd.to_numeric(frame.product_missing, errors="raise").to_numpy(dtype=np.float32)
        if not np.isin(product_missing, [0., 1.]).all():
            raise ValueError("product_missing must contain only 0 or 1")
    assay = np.stack([ph, pm, temp, tm, product_missing,
                      [cache[x][1] for x in left], [cache[x][1] for x in right]], axis=1).astype(np.float32)
    matrix = sparse.hstack([sparse.csr_matrix(np.stack([cache[x][0] for x in side]))
                            for side in (left,right)] + [sparse.csr_matrix(assay)], format="csr", dtype=np.float32)
    names = [f"{side}_morgan_{i:04d}" for side in ("reactant", "product") for i in range(morgan_bits)]
    names += ["assay_ph", "assay_ph_missing", "assay_temperature", "assay_temperature_missing",
              ("assay_product_missing" if use_product_missing else "assay_reserved"),
              "assay_reactant_valid", "assay_product_valid"]
    groups = [("reactant_morgan",0,morgan_bits), ("product_morgan",morgan_bits,2*morgan_bits),
              ("assay_numeric",2*morgan_bits,2*morgan_bits+7)]
    return matrix, names, groups, {"rows": len(frame), "features": matrix.shape[1], "unique_smiles": len(cache)}

def _load_or_build_features(
    frame: pd.DataFrame,
    *,
    output_dir: Path,
    protocol: dict[str, Any],
    morgan_bits: int,
    force_rebuild: bool,
    use_product_missing: bool = False,
) -> tuple[sparse.csr_matrix, list[str], list[tuple[str, int, int]], dict[str, Any]]:
    import os
    output_dir.mkdir(parents=True, exist_ok=True)
    lock = output_dir / '.BUILDING.lock'
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(f'Handcrafted cache is in use: {lock}. Check the PID before removing a stale lock.') from error
    try:
        with os.fdopen(descriptor, 'w') as stream:
            stream.write(str(os.getpid()))
        return _load_or_build_features_locked(frame, output_dir=output_dir, protocol=protocol,
            morgan_bits=morgan_bits, force_rebuild=force_rebuild, use_product_missing=use_product_missing)
    finally:
        lock.unlink(missing_ok=True)


def _load_or_build_features_locked(frame, *, output_dir, protocol, morgan_bits, force_rebuild,
                                   use_product_missing=False):
    output_dir.mkdir(parents=True, exist_ok=True)
    implementation = {'rdkit_version': version('rdkit'), 'implementation_sha256': _sha256(Path(__file__))}
    cache_path = output_dir / "handcrafted_features.npz"
    manifest_path = output_dir / "feature_manifest.json"
    if cache_path.is_file() and manifest_path.is_file() and not force_rebuild:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            **implementation,
            "dataset_sha256": protocol["dataset_sha256"],
            "morgan_bits": morgan_bits,
            "feature_version": ("minimal_morgan_product_missing_1" if use_product_missing
                                else "minimal_morgan_1"),
        }
        if use_product_missing:
            if not protocol.get("product_missing_source_sha256"):
                raise ValueError("Product missingness requires original source SHA256")
            expected["product_missing_source_sha256"] = protocol["product_missing_source_sha256"]
        actual = {key: manifest.get(key) for key in expected}
        if actual != expected:
            raise RuntimeError(
                f"Existing feature cache is incompatible: expected={expected}, actual={actual}. "
                "Archive the handcrafted_features.npz and feature_manifest.json files together, then rebuild."
            )
        matrix = sparse.load_npz(cache_path).tocsr().astype(np.float32)
        names = list(manifest["feature_names"])
        groups = [(item["name"], int(item["start"]), int(item["end"])) for item in manifest["groups"]]
        if matrix.shape != (len(frame), len(names)):
            raise RuntimeError(f"Cached feature shape mismatch: {matrix.shape}")
        print(f"Resume handcrafted feature cache: {cache_path} shape={matrix.shape}")
        return matrix, names, groups, dict(manifest["stats"])

    if use_product_missing and not protocol.get("product_missing_source_sha256"):
        raise ValueError("Product missingness requires original source SHA256")
    matrix, names, groups, stats = _build_feature_matrix(
        frame, morgan_bits=morgan_bits, use_product_missing=use_product_missing)
    from ..runtime import atomic_output
    with atomic_output(cache_path) as temporary:
        sparse.save_npz(temporary, matrix, compressed=True)
    manifest = {
        **implementation,
        "feature_version": ("minimal_morgan_product_missing_1" if use_product_missing
                            else "minimal_morgan_1"),
        "dataset_sha256": protocol["dataset_sha256"],
        "morgan_radius": 2,
        "morgan_bits": morgan_bits,
        "feature_names": names,
        "groups": [{"name": name, "start": start, "end": end} for name, start, end in groups],
        "stats": stats,
        "excluded_identity_fields": ["ec", "taxonomy_id", "uniprot", "cofactor", "sequence_source", "group", "sample_weight"],
        "uses_pretrained_embeddings": False,
        "uses_supervised_dl_features": False,
    }
    if use_product_missing:
        manifest.update(product_missing_source_sha256=protocol["product_missing_source_sha256"],
                        product_missing_rows=int(frame.product_missing.sum()),
                        product_missing_policy="Original product string empty before reactant fallback; 1=missing")
    _write_json(manifest_path, manifest)
    print(f"Wrote handcrafted feature cache: {cache_path} shape={matrix.shape}")
    return matrix, names, groups, stats

