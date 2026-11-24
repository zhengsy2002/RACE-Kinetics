from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace
from .config import validate_paths


def training_args(config):
    data, output = config["data"], Path(config["output_dir"])
    return SimpleNamespace(**config["training"],
        direction_policy=config["supervision"]["direction"],
        **{f"{key}_weight": value for key, value in config["loss"].items()},
        prepared_dir=Path(data["prepared_dir"]), protein_cache_dir=Path(data["protein_pretrain_cache"]),
        molformer_cache_dir=Path(data["molformer_cache"]), output_dir=output / "weights",
        dropout=config["model"]["dropout"], protein_projection=config["model"]["protein_projection"],
        protein_bottleneck=config["model"]["protein_bottleneck"],
        )


def extract_features(config):

    validate_paths(config)
    from .backbones.download import ensure_models
    from .utils import bind_files
    from .backbones import esmc, molformer
    data, options = config["data"], config["extraction"]
    device = config["training"]["device"]
    for dataset, key in (("reactzyme", "protein_pretrain_cache"), ("kcat", "protein_downstream_cache")):
        folder = Path(data[key])
        meta_path = folder / "metadata.json"
        if meta_path.is_file() and json.loads(meta_path.read_text(encoding="utf-8")).get("complete"):
            print(f"Reuse complete ESM-C cache (identity audited below): {folder}", flush=True)
            continue
        ensure_models(config, ("esmc",))
        model_dir = Path(data["esmc_model"])
        args = SimpleNamespace(dataset=dataset, kcat_csv=Path(data["raw_csv"]),
            prepared_dir=Path(data["prepared_dir"]), protein_tsv=Path(data["protein_tsv"]),
            output_dir=folder, model_dir=model_dir, device=device, backend="transformers",
            model_dtype=options["model_dtype"], max_residues=options["max_residues"],
            overlap=options["overlap"], flush_every=options["flush_every"], max_sequences=0)
        identity = folder / "extraction_identity.json"
        if folder.exists() and any(p.name != ".gitkeep" for p in folder.iterdir()) and not identity.exists():
            raise ValueError(f"Incomplete unowned protein cache; choose a new cache directory: {folder}")
        files = [Path(data["raw_csv"])] if dataset == "kcat" else [
            Path(data["protein_tsv"]), Path(data["prepared_dir"]) / "protein_index.tsv"]
        files += model_files(model_dir)
        bind_files(identity, files, {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
        esmc.extract(args)
    folder = Path(data["molformer_cache"])
    meta_path = folder / "metadata.json"
    if meta_path.is_file() and json.loads(meta_path.read_text(encoding="utf-8")).get("complete"):
        print(f"Reuse complete MolFormer cache (identity audited below): {folder}", flush=True)
        return
    ensure_models(config, ("molformer",))
    model_dir = Path(data["molformer_model"])
    identity = folder / "extraction_identity.json"
    if folder.exists() and any(p.name != ".gitkeep" for p in folder.iterdir()) and not identity.exists():
        raise ValueError(f"Incomplete unowned MolFormer cache; choose a new directory: {folder}")
    bind_files(identity, model_files(model_dir), options)
    molformer.extract(SimpleNamespace(prepared_dir=Path(data["prepared_dir"]),
        kcat_csv=Path(data["downstream_csv"]), output_dir=folder, model=str(model_dir), revision=None,
        device=device, batch_size=options["batch_size"], max_length=options["max_length"],
        embedding_dim=768, max_components=0, flush_every=options["flush_every"]))


def model_files(folder):
    if not (folder / "config.json").is_file():
        raise FileNotFoundError(f"Local Hugging Face model config missing: {folder}")
    files = sorted(p for p in folder.rglob("*") if p.is_file() and not {".git", ".cache"}.intersection(p.parts)
                   and p.suffix in {".json", ".py", ".txt", ".bin", ".safetensors", ".model"})
    if not any(p.suffix in {".bin", ".safetensors"} for p in files):
        raise FileNotFoundError(f"Model weights missing: {folder}")
    return files


