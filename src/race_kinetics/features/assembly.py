import json
from pathlib import Path
import numpy as np
import torch
from scipy import sparse
from .alignment import _load_checkpoint, _build_or_load_cache, _sample_features
from ..utils import freeze_json, write_json, sha256


def select_assay_features(matrix, names, groups, drop_assay):
    removed = {f"assay_{key}{suffix}" for key in drop_assay for suffix in ("", "_missing")}
    keep = np.array([i for i, name in enumerate(names) if name not in removed], dtype=int)
    selected_groups, offset = [], 0
    for name, start, end in groups:
        width = int(((keep >= start) & (keep < end)).sum())
        if width:
            selected_groups.append((name, offset, offset + width))
            offset += width
    return matrix[:, keep].tocsr(), [names[i] for i in keep], selected_groups


def combine_features(handcrafted, learned, handcrafted_groups, learned_groups, mode):
    if mode == "handcrafted_only":
        return handcrafted.tocsr(), list(handcrafted_groups)
    if mode == "learned_only":
        return sparse.csr_matrix(learned), list(learned_groups)
    if mode != "all":
        raise ValueError(f"Unknown feature mode: {mode}")
    offset = handcrafted.shape[1]
    groups = list(handcrafted_groups) + [(k, a + offset, b + offset) for k, a, b in learned_groups]
    return sparse.hstack([handcrafted, sparse.csr_matrix(learned)], format="csr"), groups


def learned_features(config, frame, checkpoint_path, folder):
    folder.mkdir(parents=True, exist_ok=True)
    data = config["data"]
    freeze_json(folder / "learned_identity.json", {
        "dataset_sha256": sha256(Path(data["downstream_csv"])),
        "checkpoint_sha256": sha256(checkpoint_path),
        "feature_layout": "substrate_enzyme_reaction_no_interactions_768"})
    array_path, groups_path = folder / "sample_features.npy", folder / "sample_groups.json"
    if array_path.exists() and groups_path.exists():
        return np.load(array_path, mmap_mode="r"), json.loads(groups_path.read_text())
    checkpoint = _load_checkpoint(checkpoint_path)
    device = torch.device(config["training"]["device"])
    batch_size = config["evaluation"]["batch_size"]
    proteins, _, reactions, manifest, rows, _ = _build_or_load_cache(
        frame, data_path=Path(data["downstream_csv"]),
        sequence_cache_dir=Path(data["protein_downstream_cache"]),
        molformer_cache_dir=Path(data["molformer_cache"]),
        checkpoint_path=checkpoint_path, checkpoint=checkpoint, cache_dir=folder / "alignment",
        device=device, batch_size=batch_size, force_rebuild=False)
    matrix, groups = _sample_features(frame, proteins, rows, reactions, manifest,
        molformer_cache_dir=Path(data["molformer_cache"]), checkpoint=checkpoint,
        device=device, batch_size=batch_size)
    from ..runtime import atomic_output
    with atomic_output(array_path) as temp:
        with temp.open("wb") as handle:
            np.save(handle, matrix)
    write_json(groups_path, groups)
    return matrix, groups


