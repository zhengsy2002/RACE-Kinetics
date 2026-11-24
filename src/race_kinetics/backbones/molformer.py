from __future__ import annotations
import json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from ..data.protocol import _reaction_roles
from ..utils import sha256 as _sha256, write_json as _write_json
def _ordered_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))


def _components(side: str) -> list[str]:
    values = [value.strip() for value in str(side).split(".") if value.strip()]
    if not values:
        raise ValueError(f"Reaction side contains no molecular component: {side!r}")
    return values


def _pooled_output(outputs: object, attention_mask: torch.Tensor) -> torch.Tensor:
    pooled = getattr(outputs, "pooler_output", None)
    if pooled is not None:
        return pooled
    hidden = getattr(outputs, "last_hidden_state", None)
    if hidden is None:
        raise RuntimeError("MolFormer output has neither pooler_output nor last_hidden_state")
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


def extract(args) -> None:
    reaction_path = args.prepared_dir / "reaction_index.tsv"
    for path in (reaction_path, args.kcat_csv):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.batch_size <= 0 or args.max_length <= 0 or args.embedding_dim <= 0:
        raise ValueError("batch-size, max-length and embedding-dim must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    reactions = pd.read_csv(reaction_path, sep="\t").fillna("")
    kcat = pd.read_csv(args.kcat_csv).fillna("")
    kcat_reactants, kcat_products = _reaction_roles(kcat)
    sides = _ordered_unique(
        reactions["substrate"].astype(str).tolist()
        + reactions["product"].astype(str).tolist()
        + kcat["reactant_smiles"].astype(str).tolist()
        + kcat_reactants
        + kcat_products
    )
    components = _ordered_unique(
        [component for side in sides for component in _components(side)]
    )
    if args.max_components > 0:
        components = components[: args.max_components]
        allowed = set(components)
        sides = [side for side in sides if set(_components(side)).issubset(allowed)]
    if not sides or not components:
        raise RuntimeError("No MolFormer inputs were collected")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    component_manifest_path = args.output_dir / "unique_components.csv"
    side_manifest_path = args.output_dir / "unique_sides.csv"
    component_path = args.output_dir / "component_embeddings.npy"
    completed_path = args.output_dir / "component_completed.npy"
    side_path = args.output_dir / "side_embeddings.npy"
    config_path = args.output_dir / "cache_config.json"
    metadata_path = args.output_dir / "metadata.json"
    config = {
        "format": "race_kinet_molecule_cache",
        "reactzyme_reaction_index": str(reaction_path.resolve()),
        "reactzyme_reaction_index_sha256": _sha256(reaction_path),
        "kcat_csv": str(args.kcat_csv.resolve()),
        "kcat_csv_sha256": _sha256(args.kcat_csv),
        "model": args.model,
        "revision": args.revision or None,
        "pooling": "MolFormer pooler_output per dot-separated component; arithmetic mean per side",
        "max_length": args.max_length,
        "embedding_dim": args.embedding_dim,
        "unique_components": len(components),
        "unique_sides": len(sides),
        "uses_kcat_labels": False,
    }
    if config_path.is_file():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != config:
            raise RuntimeError(
                "Existing MolFormer cache uses different sources/settings; choose a new output directory"
            )
    else:
        _write_json(config_path, config)
        pd.DataFrame(
            {"cache_row": np.arange(len(components)), "component_smiles": components}
        ).to_csv(component_manifest_path, index=False)
        pd.DataFrame(
            {
                "cache_row": np.arange(len(sides)),
                "side_smiles": sides,
                "component_count": [len(_components(side)) for side in sides],
            }
        ).to_csv(side_manifest_path, index=False)

    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("complete", False) and metadata.get("config") == config:
            print(f"MolFormer cache already complete: {args.output_dir}")
            return

    if component_path.is_file():
        component_embeddings = np.load(component_path, mmap_mode="r+")
        completed = np.load(completed_path, mmap_mode="r+")
        if component_embeddings.shape != (len(components), args.embedding_dim):
            raise RuntimeError(f"Incompatible component cache: {component_embeddings.shape}")
        if completed.shape != (len(components),):
            raise RuntimeError(f"Incompatible completion cache: {completed.shape}")
    else:
        component_embeddings = np.lib.format.open_memmap(
            component_path,
            mode="w+",
            dtype=np.float16,
            shape=(len(components), args.embedding_dim),
        )
        completed = np.lib.format.open_memmap(
            completed_path, mode="w+", dtype=np.bool_, shape=(len(components),)
        )
        completed[:] = False
        component_embeddings.flush()
        completed.flush()

    pending = np.flatnonzero(~np.asarray(completed, dtype=bool))
    if len(pending):
        from transformers import AutoModel, AutoTokenizer

        load_kwargs: dict[str, object] = {"trust_remote_code": True, "local_files_only": True}
        if args.revision:
            load_kwargs["revision"] = args.revision
        tokenizer = AutoTokenizer.from_pretrained(args.model, **load_kwargs)
        model = AutoModel.from_pretrained(
            args.model, deterministic_eval=True, **load_kwargs
        ).eval().to(args.device)
        progress = tqdm(total=len(pending), desc="Frozen MolFormer", unit="component")
        truncated = 0
        for batch_index, start in enumerate(range(0, len(pending), args.batch_size)):
            rows = pending[start : start + args.batch_size]
            smiles = [components[int(row)] for row in rows]
            raw_ids = tokenizer(smiles, padding=False, truncation=False)["input_ids"]
            truncated += sum(len(ids) > args.max_length for ids in raw_ids)
            encoded = tokenizer(
                smiles,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(args.device) for key, value in encoded.items()}


            with torch.inference_mode():
                pooled = _pooled_output(model(**encoded), encoded["attention_mask"])
            values = pooled.float().cpu().numpy()
            if values.shape != (len(rows), args.embedding_dim) or not np.isfinite(values).all():
                nonfinite = int((~np.isfinite(values)).sum())
                raise RuntimeError(
                    "Invalid MolFormer batch output: "
                    f"shape={values.shape}, expected={(len(rows), args.embedding_dim)}, "
                    f"nonfinite={nonfinite}"
                )
            component_embeddings[rows] = values.astype(np.float16)
            completed[rows] = True
            if (batch_index + 1) % args.flush_every == 0:
                component_embeddings.flush()
                completed.flush()
            progress.update(len(rows))
        progress.close()
        component_embeddings.flush()
        completed.flush()
    else:
        truncated = 0

    if not np.asarray(completed, dtype=bool).all():
        raise RuntimeError("MolFormer component cache is incomplete")
    component_lookup = {value: row for row, value in enumerate(components)}
    side_embeddings = np.lib.format.open_memmap(
        side_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(sides), args.embedding_dim),
    )
    for row, side in enumerate(tqdm(sides, desc="Aggregate reaction sides", unit="side")):
        indices = [component_lookup[value] for value in _components(side)]
        value = np.asarray(component_embeddings[indices], dtype=np.float32).mean(axis=0)
        if not np.isfinite(value).all():
            raise FloatingPointError(f"Non-finite MolFormer side embedding at row {row}")
        side_embeddings[row] = value.astype(np.float16)
    side_embeddings.flush()
    metadata = {
        "format": config["format"],
        "complete": True,
        "config": config,
        "component_embeddings_shape": list(component_embeddings.shape),
        "side_embeddings_shape": list(side_embeddings.shape),
        "truncated_components_in_final_resume_segment": int(truncated),
        "dtype": "float16",
        "input_key": "exact_side_smiles",
    }
    _write_json(metadata_path, metadata)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
