from pathlib import Path

import numpy as np
from scipy import sparse

from .handcrafted import _load_or_build_features, _validate_protocol
from ..data.kcat import attach_product_missing
from .assembly import select_assay_features, combine_features, learned_features
from ..storage import handcrafted_dir, aligned_dir, DATA_ROOT, content_key
from ..utils import sha256, freeze_json


def build_matrix(config, frame, checkpoint):
    mode = config["features"]["mode"]
    missing = config["features"].get("product_missing", False)
    if missing:
        frame = attach_product_missing(frame, config["data"]["raw_csv"])
    hand, learned, hg, lg = None, None, [], []
    sources = []
    if mode != "learned_only":
        protocol = _validate_protocol(frame, Path(config["data"]["downstream_csv"]))
        if missing:
            protocol["product_missing_source_sha256"] = sha256(Path(config["data"]["raw_csv"]))
        folder = handcrafted_dir(config)
        hand, names, hg, _ = _load_or_build_features(frame, output_dir=folder,
            protocol=protocol, morgan_bits=1024, force_rebuild=False, use_product_missing=missing)
        hand, names, hg = select_assay_features(hand, names, hg, config["features"]["drop_assay"])
        sources += [folder / "handcrafted_features.npz", folder / "feature_manifest.json"]
    if mode != "handcrafted_only":
        folder = aligned_dir(config, checkpoint)
        learned, lg = learned_features(config, frame, checkpoint, folder)
        sources += [folder / "sample_features.npy", folder / "sample_groups.json", checkpoint]
    matrix, groups = combine_features(hand, learned, hg, lg, mode)
    expected = {"all": 2823, "learned_only": 768, "handcrafted_only": 2055}[mode]
    if mode != "learned_only":
        expected -= 2 * len(config["features"]["drop_assay"])
    if matrix.shape != (len(frame), expected) or not np.isfinite(matrix.data).all():
        raise ValueError(f"Invalid feature matrix: {matrix.shape}, expected {(len(frame), expected)}")
    identity = {"dataset": sha256(Path(config["data"]["downstream_csv"])),
                "features": config["features"], "sources": {str(p): sha256(p) for p in sources}}
    folder = DATA_ROOT / "features/matrices" / content_key(identity)
    folder.mkdir(parents=True, exist_ok=True)
    freeze_json(folder / "identity.json", identity)
    path = folder / "features.npz"
    if path.exists():
        existing = sparse.load_npz(path)
        if existing.shape != matrix.shape or (existing != matrix).nnz:
            raise ValueError("Saved feature matrix differs from reconstructed inputs")
    else:
        sparse.save_npz(path, matrix)
    freeze_json(folder / "groups.json", groups)
    return matrix, path
