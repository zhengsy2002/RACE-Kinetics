from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from .caches import (_normalize_sequence, _sequence_sha1, _validate_sequence_cache,
                     validate_protein_metadata, load_side_cache)
from .protocol import _reaction_roles
from ..features.handcrafted import _validate_protocol
from ..utils import sha256


def reaction_key(left, right):
    def side(value):
        if pd.isna(value):
            return ""
        parts = []
        for text in str(value).strip().split("."):
            text = text.strip()
            if not text or text.lower() == "nan":
                continue
            try:
                molecule = Chem.MolFromSmiles(text)
                parts.append(Chem.MolToSmiles(molecule, canonical=True) if molecule is not None else text)
            except Exception:
                parts.append(text)
        return ".".join(sorted(parts))
    a, b = side(left), side(right)
    return "||".join(sorted((a, b))) if a and b else None


def finite(array):
    for start in range(0, len(array), 4096):
        if not np.isfinite(array[start:start + 4096]).all():
            raise ValueError("Nonfinite embedding values")


def audit_data(data):
    data = {k: Path(v) for k, v in data.items()}
    from .kcat import validate_derived
    validate_derived(data["raw_csv"], data["downstream_csv"])
    raw = pd.read_csv(data["raw_csv"], dtype={"sample_id": str})
    frame = pd.read_csv(data["downstream_csv"], dtype={"sample_id": str})
    protocol = _validate_protocol(frame, data["downstream_csv"])
    if (not raw.sample_id.is_unique or raw.sample_id.isna().any()
            or frame["sample_id"].isna().any()
            or raw.sample_id.tolist() != frame["sample_id"].tolist()):
        raise ValueError("Raw/processed dataset sample IDs or row order differ")
    if not np.allclose(raw.log10_value, frame.log10_value, rtol=0, atol=1e-7):
        raise ValueError("Original labels changed")
    if not np.array_equal(raw.sequence.map(_normalize_sequence), frame.sequence.map(_normalize_sequence)):
        raise ValueError("Raw/processed protein sequences differ")
    reactants = raw.reactant_smiles.fillna("").str.strip()
    products = raw.product_smiles.fillna("").str.strip()
    if (not np.array_equal(frame.reactant_smiles.fillna("").str.strip(), reactants)
            or not np.array_equal(frame.product_smiles.fillna("").str.strip(), products.where(products.ne(""), reactants))):
        raise ValueError("Reaction fields differ from the established missing-product fallback")
    prepared = data["prepared_dir"]
    proteins = pd.read_csv(prepared / "protein_index.tsv", sep="\t")
    reactions = pd.read_csv(prepared / "reaction_index.tsv", sep="\t").fillna("")
    pairs = pd.read_csv(prepared / "pairs.tsv", sep="\t")
    for table, key in ((proteins, "protein_row"), (reactions, "reaction_row")):
        if table.empty or table[key].tolist() != list(range(len(table))):
            raise ValueError(f"{key} must be nonempty, contiguous and ordered")
    if (pairs.empty or not pairs.protein_row.isin(proteins.protein_row).all()
            or not pairs.reaction_row.isin(reactions.reaction_row).all()):
        raise ValueError("Invalid pretraining pair indices")
    if (set(pairs.split) != {"train", "valid"} or pairs.rhea_id.isna().any()
            or (pairs.groupby("rhea_id").split.nunique() > 1).any()
            or (pairs.groupby("reaction_row").split.nunique() > 1).any()):
        raise ValueError("Pretraining train/valid reactions are not disjoint")
    protein_overlap = set(raw.sequence.map(_sequence_sha1)) & set(proteins.sequence_sha1)
    target_keys = {reaction_key(r, p) for r, p in zip(raw.reactant_smiles, raw.product_smiles)} - {None}
    reaction_overlap = [i for i, (r, p) in enumerate(zip(reactions.substrate, reactions["product"]))
                        if reaction_key(r, p) in target_keys]
    if protein_overlap or reaction_overlap:
        raise ValueError(f"Target/pretraining overlap: {len(protein_overlap)} proteins, {len(reaction_overlap)} reactions")
    stats = json.loads((prepared / "prepare_stats.json").read_text(encoding="utf-8"))
    if stats.get("protocol") != "kcat-only exact pretraining exclusion" or stats.get("kcat_rows") != len(raw):
        raise ValueError("Prepared provenance does not match raw dataset")
    if stats.get("raw_sha256") != sha256(data["raw_csv"]):
        raise ValueError("Prepared target SHA256 does not match raw dataset")
    protocol.update(pretraining_pairs=len(pairs), pretraining_proteins=len(proteins),
                    pretraining_reactions=len(reactions), target_protein_overlap=0, target_reaction_overlap=0)
    return frame, proteins, reactions, protocol


def audit_proteins(data, frame, proteins):
    files, metadata = [], []
    for key, mapping_name in (("protein_pretrain_cache", "protein_row_to_cache_row.npy"),
                              ("protein_downstream_cache", "sample_row_to_cache_row.npy")):
        folder = Path(data[key])
        files += [folder / name for name in ("metadata.json", "unique_sequences.csv", "unique_embeddings.npy", mapping_name)]
        meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        validate_protein_metadata(meta)
        metadata.append(meta)
        if key == "protein_downstream_cache":
            array, _, _, _ = _validate_sequence_cache(frame, folder)
        else:
            manifest = pd.read_csv(folder / "unique_sequences.csv")
            mapping = np.load(folder / mapping_name)
            array = np.load(folder / "unique_embeddings.npy", mmap_mode="r")
            if (manifest.cache_row.tolist() != list(range(len(manifest))) or not manifest.sequence_sha1.is_unique
                    or mapping.dtype.kind not in "iu" or mapping.shape != (len(proteins),)
                    or mapping.min() < 0 or mapping.max() >= len(manifest)
                    or array.shape != (len(manifest), 1152)
                    or not np.array_equal(manifest.sequence_sha1.to_numpy()[mapping], proteins.sequence_sha1)):
                raise ValueError("Pretraining sequence cache mapping mismatch")
        finite(array)
    for key in ("hidden_size", "backbone", "backend", "model_dir", "pooling", "model_dtype", "architecture"):
        if metadata[0].get(key) != metadata[1].get(key):
            raise ValueError(f"Pretraining/downstream ESM-C caches differ: {key}")
    return files


def audit_molecules(data, frame, reactions):
    folder = Path(data["molformer_cache"])
    array, lookup, metadata = load_side_cache(folder)
    finite(array)
    config = metadata["config"]
    if not config.get("model"):
        raise ValueError("MolFormer cache has no original model identity")

    if str(config["model"]) != str(Path(data["molformer_model"]).resolve()):
        raise ValueError("MolFormer cache model path differs; extract a fresh cache")
    if array.shape[1] != 768:
        raise ValueError("MolFormer requires 768-dimensional embeddings")
    for key, source in (("kcat_csv_sha256", Path(data["downstream_csv"])),
                        ("reactzyme_reaction_index_sha256", Path(data["prepared_dir"]) / "reaction_index.tsv")):
        if config.get(key) != sha256(source):
            raise ValueError(f"MolFormer cache source changed: {key}")
    left, right = _reaction_roles(frame)
    required = set(left + right + frame.reactant_smiles.fillna("").astype(str).tolist()
                   + reactions.substrate.astype(str).tolist() + reactions["product"].astype(str).tolist())
    if not required.issubset(lookup):
        raise ValueError("MolFormer cache does not cover current molecule sides")
    return [folder / name for name in ("metadata.json", "unique_sides.csv", "side_embeddings.npy")]
