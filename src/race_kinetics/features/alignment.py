from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from ..models.alignment import MolFormerAdapter, ReactionAlignmentModel
from ..models.projections import checkpoint_model_options
from ..data.caches import load_side_cache, _validate_sequence_cache
from ..data.protocol import _reaction_roles
from ..utils import sha256 as _sha256, write_json as _write_json, _signature
def _load_checkpoint(path: Path) -> dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "format",
        "protein_projection",
        "molecule_adapter",
        "reaction_projection",
        "molformer_dim",
    }
    missing = sorted(required.difference(checkpoint))
    if checkpoint.get("format") != "race_kinetics_alignment" or missing:
        raise ValueError(
            f"Unsupported RACE-Kinet checkpoint: format={checkpoint.get('format')!r}, "
            f"missing={missing}"
        )
    checkpoint_model_options(checkpoint)
    return checkpoint


def _assert_pretraining_complete(path: Path) -> dict[str, object]:
    checkpoint = _load_checkpoint(path)
    completion = checkpoint.get("training_completion")
    if not isinstance(completion, dict):
        raise RuntimeError("Checkpoint lacks embedded training completion metadata")
    epoch = int(completion["last_epoch"])
    expected = int(completion["configured_epochs"])
    bad = int(completion["bad_epochs"])
    patience = int(completion["early_stopping_patience"])
    if not ((expected > 0 and epoch >= expected) or (patience > 0 and bad >= patience)):
        raise RuntimeError("Pretraining has not completed")
    return {**completion, "complete": True}


def _load_modules(
    checkpoint: dict[str, object], device: torch.device
) -> tuple[nn.Module, MolFormerAdapter, nn.Sequential]:
    architecture = ReactionAlignmentModel(**checkpoint_model_options(checkpoint))
    protein = architecture.protein_projection
    protein.load_state_dict(checkpoint["protein_projection"], strict=True)
    molecule = architecture.molecule_adapter
    molecule.load_state_dict(checkpoint["molecule_adapter"], strict=True)
    reaction = architecture.reaction_projection
    reaction.load_state_dict(checkpoint["reaction_projection"], strict=True)
    for module in (protein, molecule, reaction):
        module.eval().to(device)
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return protein, molecule, reaction


def _project_batches(
    values: np.ndarray,
    module: nn.Module,
    *,
    device: torch.device,
    batch_size: int,
    description: str,
) -> np.ndarray:
    output = np.empty((len(values), 256), dtype=np.float32)
    for start in tqdm(range(0, len(values), batch_size), desc=description, unit="batch"):
        rows = np.arange(start, min(start + batch_size, len(values)))
        batch = torch.from_numpy(np.asarray(values[rows], dtype=np.float32)).to(device)
        with torch.inference_mode():
            projected = module(batch)
        output[start : start + len(rows)] = projected.cpu().numpy()
    return output


def _build_or_load_cache(
    frame: pd.DataFrame,
    *,
    data_path: Path,
    sequence_cache_dir: Path,
    molformer_cache_dir: Path,
    checkpoint_path: Path,
    checkpoint: dict[str, object],
    cache_dir: Path,
    device: torch.device,
    batch_size: int,
    force_rebuild: bool,
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, dict[str, object]]:
    esm, sample_rows, sequence_manifest, sequence_metadata = _validate_sequence_cache(
        frame, sequence_cache_dir
    )
    sides, side_lookup, molformer_metadata = load_side_cache(molformer_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    protein_path = cache_dir / "unique_sequence_embeddings.npy"
    protein_manifest_path = cache_dir / "unique_sequences.csv"
    reaction_path = cache_dir / "unique_reaction_embeddings.npy"
    reaction_manifest_path = cache_dir / "unique_reactions.csv"
    metadata_path = cache_dir / "metadata.json"
    identity = {
        "feature_version": 1,
        "dataset_sha256": _sha256(data_path),
        "sequence_cache_metadata": _signature(sequence_cache_dir / "metadata.json"),
        "sequence_row_mapping": _signature(sequence_cache_dir / "sample_row_to_cache_row.npy"),
        "molformer_cache_metadata": _signature(molformer_cache_dir / "metadata.json"),
        "crossmodal_checkpoint": _signature(checkpoint_path),
        "checkpoint_format": checkpoint["format"],
        "checkpoint_architecture": checkpoint_model_options(checkpoint),
        "protein_key": "normalized_sequence_sha1",
        "protein_dim": 256,
        "reaction_dim": 256,
        "normalization": "L2 after frozen RACE-Kinet projection",
    }
    files = (
        protein_path,
        protein_manifest_path,
        reaction_path,
        reaction_manifest_path,
        metadata_path,
    )
    if all(path.is_file() for path in files) and not force_rebuild:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if {key: metadata.get(key) for key in identity} != identity or not metadata.get(
            "complete", False
        ):
            raise RuntimeError("Existing RACE-Kinet downstream cache is incompatible")
        proteins = np.load(protein_path, mmap_mode="r")
        protein_manifest = pd.read_csv(protein_manifest_path, dtype=str).fillna("")
        reactions = np.load(reaction_path, mmap_mode="r")
        reaction_manifest = pd.read_csv(reaction_manifest_path, keep_default_na=False)
        return proteins, protein_manifest, reactions, reaction_manifest, sample_rows, metadata

    reactants, products = _reaction_roles(frame)
    unique_reactions = list(dict.fromkeys(zip(reactants, products)))
    try:
        reactant_rows = np.asarray(
            [side_lookup[value] for value, _ in unique_reactions], dtype=np.int64
        )
        product_rows = np.asarray(
            [side_lookup[value] for _, value in unique_reactions], dtype=np.int64
        )
    except KeyError as exc:
        raise KeyError(f"MolFormer cache does not cover kcat reaction side: {exc}") from exc
    protein_module, molecule_module, reaction_module = _load_modules(checkpoint, device)
    proteins = _project_batches(
        esm,
        protein_module,
        device=device,
        batch_size=batch_size,
        description="RACE-Kinet proteins",
    )
    raw_reactants = _project_batches(
        sides[reactant_rows],
        molecule_module,
        device=device,
        batch_size=batch_size,
        description="RACE-Kinet reactants",
    )
    raw_products = _project_batches(
        sides[product_rows],
        molecule_module,
        device=device,
        batch_size=batch_size,
        description="RACE-Kinet products",
    )
    reaction_input = np.concatenate(
        [raw_reactants, raw_products, raw_products - raw_reactants], axis=1
    )
    reactions = _project_batches(
        reaction_input,
        reaction_module,
        device=device,
        batch_size=batch_size,
        description="RACE-Kinet reactions",
    )
    proteins /= np.linalg.norm(proteins, axis=1, keepdims=True).clip(min=1e-8)
    reactions /= np.linalg.norm(reactions, axis=1, keepdims=True).clip(min=1e-8)
    protein_manifest = sequence_manifest[["cache_row", "sequence_sha1"]].copy()
    reaction_manifest = pd.DataFrame(
        {
            "cache_row": np.arange(len(unique_reactions)),
            "reactant_smiles": [item[0] for item in unique_reactions],
            "product_smiles": [item[1] for item in unique_reactions],
        }
    )
    np.save(protein_path, proteins)
    np.save(reaction_path, reactions)
    protein_manifest.to_csv(protein_manifest_path, index=False)
    reaction_manifest.to_csv(reaction_manifest_path, index=False)
    metadata = {
        **identity,
        "complete": True,
        "num_unique_sequences": len(protein_manifest),
        "num_unique_reactions": len(reaction_manifest),
        "uses_kcat_labels": False,
        "encoder_training_on_kcat": False,
        "sequence_cache": sequence_metadata,
        "molformer_cache": molformer_metadata,
        "feature_definition": (
            "Frozen component-aware MolFormer side embeddings; trainable RACE-Kinet adapter learned only "
            "on kcat-excluded ReactZyme; identical 768-dimensional downstream layout to RACE-Kinet."
        ),
    }
    _write_json(metadata_path, metadata)
    return proteins, protein_manifest, reactions, reaction_manifest, sample_rows, metadata


def _sample_features(
    frame: pd.DataFrame,
    proteins: np.ndarray,
    protein_rows: np.ndarray,
    reactions: np.ndarray,
    reaction_manifest: pd.DataFrame,
    *,
    molformer_cache_dir: Path,
    checkpoint: dict[str, object],
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, list[tuple[str, int, int]]]:
    reactants, products = _reaction_roles(frame)
    lookup = {
        (str(reactant), str(product)): int(row)
        for reactant, product, row in zip(
            reaction_manifest["reactant_smiles"],
            reaction_manifest["product_smiles"],
            reaction_manifest["cache_row"],
        )
    }
    reaction_rows = np.asarray(
        [lookup[(reactant, product)] for reactant, product in zip(reactants, products)],
        dtype=np.int64,
    )
    side_embeddings, side_lookup, _ = load_side_cache(molformer_cache_dir)
    try:
        substrate_rows = np.asarray(
            [side_lookup[str(value)] for value in frame["reactant_smiles"].fillna("")], dtype=np.int64
        )
    except KeyError as exc:
        raise KeyError(f"MolFormer cache does not cover kcat substrate: {exc}") from exc
    _, molecule_module, _ = _load_modules(checkpoint, device)
    unique_substrate_rows, inverse = np.unique(substrate_rows, return_inverse=True)
    unique_substrates = _project_batches(
        side_embeddings[unique_substrate_rows],
        molecule_module,
        device=device,
        batch_size=batch_size,
        description="RACE-Kinet substrates",
    )
    substrate = unique_substrates[inverse]
    protein = np.asarray(proteins[protein_rows], dtype=np.float32)
    reaction = np.asarray(reactions[reaction_rows], dtype=np.float32)
    values = np.concatenate(
        [
            substrate,
            protein,
            reaction,
        ],
        axis=1,
    ).astype(np.float32, copy=False)
    if values.shape != (len(frame), 768) or not np.isfinite(values).all():
        raise RuntimeError(f"Invalid RACE-Kinet sample features: {values.shape}")
    groups: list[tuple[str, int, int]] = []
    cursor = 0
    for name, width in (
        ("molformer_substrate", 256),
        ("aligned_protein", 256),
        ("aligned_reaction", 256),
    ):
        groups.append((name, cursor, cursor + width))
        cursor += width
    return values, groups
