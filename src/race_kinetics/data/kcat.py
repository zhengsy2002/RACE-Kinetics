from pathlib import Path
import json
import numpy as np
import pandas as pd


def attach_product_missing(frame, raw_path):
    raw = pd.read_csv(raw_path, usecols=["sample_id", "product_smiles"],
                      dtype={"sample_id": str})
    ids = frame["sample_id"].astype("string")
    if (raw.sample_id.isna().any() or not raw.sample_id.is_unique
            or ids.isna().any() or not ids.is_unique):
        raise ValueError("Product missingness requires unique nonempty sample IDs")
    unknown = set(ids) - set(raw.sample_id)
    if unknown:
        raise ValueError(f"Sample IDs absent from original kcat table: {sorted(unknown)[:5]}")
    missing = raw.product_smiles.fillna("").astype(str).str.strip().eq("").astype(np.uint8)
    by_id = pd.Series(missing.to_numpy(), index=raw.sample_id)
    values = by_id.loc[ids.tolist()].to_numpy(dtype=np.uint8)
    result = frame.copy()
    if "product_missing" in result:
        existing = pd.to_numeric(result.product_missing, errors="raise").to_numpy()
        if not np.array_equal(existing, values):
            raise ValueError("Existing product_missing differs from original product strings")
    result["product_missing"] = values
    return result


def native_frame(path):
    raw = pd.read_csv(path, dtype={"sample_id": str})
    required = {"sample_id", "sequence", "log10_value",
                "reactant_smiles", "product_smiles"}
    if not required.issubset(raw.columns):
        raise ValueError(f"Native kcat columns missing: {sorted(required - set(raw.columns))}")
    if raw.empty or raw.sample_id.isna().any() or not raw.sample_id.is_unique:
        raise ValueError("sample_id must be nonempty and unique")
    labels = pd.to_numeric(raw.log10_value, errors="raise")
    if not np.isfinite(labels).all():
        raise ValueError("log10_value must contain finite labels; no automatic row removal")
    sequence = raw.sequence.fillna("").astype(str).str.replace(r"\s+", "", regex=True).str.upper()
    left = raw.reactant_smiles.fillna("").astype(str).str.strip()
    right = raw.product_smiles.fillna("").astype(str).str.strip()
    if sequence.eq("").any() or left.eq("").any():
        raise ValueError("Empty sequence/reactant in kcat; no automatic row removal")
    frame = raw.copy()
    frame["sequence"] = sequence
    frame["reactant_smiles"] = left
    frame["product_smiles"] = right.where(right.ne(""), left)
    frame["log10_value"] = labels
    for key in ("ph", "temperature"):
        frame[key] = pd.to_numeric(raw.get(key, pd.Series(np.nan, index=raw.index)), errors="coerce")
    return frame


def validate_derived(raw_path, derived_path):
    expected = native_frame(raw_path)
    actual = pd.read_csv(derived_path, dtype=str).fillna("")
    if len(actual) != len(expected) or not set(expected).issubset(actual):
        raise ValueError("Derived kcat row count or columns differ from the native table")
    for key in expected:
        if pd.api.types.is_numeric_dtype(expected[key]):
            values = pd.to_numeric(actual[key].replace("", np.nan), errors="raise")
            matches = np.allclose(values, expected[key], rtol=1e-12, atol=1e-12, equal_nan=True)
        else:
            matches = np.array_equal(actual[key].to_numpy(), expected[key].fillna("").astype(str).to_numpy())
        if not matches:
            raise ValueError(f"Processed kcat field differs from native input: {key}")
    return expected


def prepare_kcat(data):
    from ..utils import sha256, write_json
    raw, derived = Path(data["raw_csv"]), Path(data["downstream_csv"])
    frame = native_frame(raw)
    derived.parent.mkdir(parents=True, exist_ok=True)
    if derived.exists():

        validate_derived(raw, derived)
    else:
        from ..storage import atomic_csv
        atomic_csv(frame, derived)
    report = {"rows": len(frame), "source_sha256": sha256(raw), "derived_sha256": sha256(derived),
              "target": "log10_value", "partition_source": "external split CSV",
              "row_filtering": False, "new_split": False, "sample_weight_used": False}
    write_json(derived.parent / "dataset_manifest.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report
