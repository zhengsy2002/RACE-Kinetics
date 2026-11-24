from __future__ import annotations

import copy
import json
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def project_path(value):
    value = str(value)
    if not value.strip() or "$" in value or "~" in value:
        raise ValueError(f"Use a project-relative path, without environment variables: {value}")
    path = Path(value)
    path = (path if path.is_absolute() else PROJECT_ROOT / path).resolve()
    if not path.is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError(f"Path must stay inside this project: {path}")
    return path


def validate_paths(config):
    values = list(config["data"].values()) + [config["output_dir"]]
    values += [config["holdout"][key] for key in ("random", "homology")]
    for value in values:
        path = project_path(value)
        if path.is_dir():
            for child in path.rglob("*"):
                if child.is_symlink() or getattr(child, "is_junction", lambda: False)():
                    project_path(child)


def configure_runtime():
    from .runtime import configure
    return configure()

DEFAULTS = {
    "downloads": {
        "auto_download": True,
        "esmc_repo_id": "biohub/ESMC-600M-hf",
        "esmc_revision": "0fb34e7e5fe1f85d0abaa3d35e2671107c0b458c",
        "molformer_repo_id": "ibm-research/MoLFormer-XL-both-10pct",
        "molformer_revision": "361063d0ad524ef77cf39b08469f6be770dc550f",
    },
    "supervision": {"direction": "conflict_masked_bce"},
    "model": {"protein_dim": 1152, "molecule_dim": 768, "dropout": 0.1,
              "protein_projection": "residual", "protein_bottleneck": 128},
    "loss": {"enzyme_reaction": 1.0, "reactant_product": 1.0, "direction": 1.0},
    "training": {"device": "cuda", "seed": 42, "epochs": 20, "batch_size": 64,
                 "num_workers": 0, "lr": 0.00005, "adapter_lr": 0.0001,
                 "new_head_lr": 0.0001, "weight_decay": 0.0001,
                 "patience": 5, "scheduler_patience": 3},
    "evaluation": {"batch_size": 128, "n_estimators": 200, "min_samples_leaf": 2,
                   "max_features": 0.3, "n_jobs": -1},
    "extraction": {"batch_size": 128, "max_length": 202, "max_residues": 2046,
                   "overlap": 128, "model_dtype": "float16", "flush_every": 50},
}
DATA_KEYS = {"raw_csv", "downstream_csv", "prepared_dir", "protein_pretrain_cache",
             "protein_downstream_cache", "molformer_cache", "protein_tsv",
             "esmc_model", "molformer_model", "reaction_tsv"}


def check_fields(value, expected, location):
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be a JSON object")
    missing, unknown = sorted(set(expected) - set(value)), sorted(set(value) - set(expected))
    if missing or unknown:
        raise ValueError(
            f"Invalid {location}: missing={missing}, unknown={unknown}. "
            "Use this project's current configs/default.json; old study/pretraining configuration is not supported."
        )


def validate(config):
    check_fields(config, set(DEFAULTS) | {"data", "output_dir", "features", "holdout"}, "top-level configuration")
    check_fields(config["data"], DATA_KEYS, "data")
    holdout = config.get("holdout", {})
    if set(holdout) != {"random", "homology", "seeds"}:
        raise ValueError("holdout requires random, homology and seeds")
    seeds = holdout["seeds"]
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds):
        raise ValueError("Training seeds must be unique nonnegative integers below 2**32")
    feature_keys = set(config.get("features", {}))
    if (not {"mode", "drop_assay"}.issubset(feature_keys)
            or feature_keys - {"mode", "drop_assay", "product_missing"}
            or config["features"]["mode"] not in ("all", "learned_only", "handcrafted_only")):
        raise ValueError("features.mode must be all, learned_only or handcrafted_only")
    if "product_missing" in config["features"] and type(config["features"]["product_missing"]) is not bool:
        raise ValueError("features.product_missing must be boolean")
    drops = config["features"]["drop_assay"]
    if not isinstance(drops, list) or len(drops) != len(set(drops)) or set(drops) - {"ph", "temperature"}:
        raise ValueError("features.drop_assay supports ph and temperature")
    if config["supervision"]["direction"] not in ("all_bce", "conflict_masked_bce"):
        raise ValueError("supervision.direction must be all_bce or conflict_masked_bce")
    for section, defaults in DEFAULTS.items():
        check_fields(config[section], defaults, section)
    downloads = config["downloads"]
    if type(downloads["auto_download"]) is not bool:
        raise ValueError("downloads.auto_download must be boolean")
    for name in ("esmc", "molformer"):
        if not isinstance(downloads[f"{name}_repo_id"], str) or not downloads[f"{name}_repo_id"].strip():
            raise ValueError(f"Missing {name} download repository")
        revision = downloads[f"{name}_revision"]
        if not isinstance(revision, str) or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise ValueError(f"downloads.{name}_revision must be a full commit SHA")
    model, loss, train, evaluation, extraction = (config[k] for k in ("model", "loss", "training", "evaluation", "extraction"))
    if (model["protein_dim"], model["molecule_dim"]) != (1152, 768):
        raise ValueError("This architecture requires ESM-C HF 1152 and MolFormer 768")
    if not 0 <= model["dropout"] < 1:
        raise ValueError("dropout must be in [0, 1)")
    if model["protein_projection"] not in ("linear", "residual"):
        raise ValueError("protein_projection must be linear or residual")
    if type(model["protein_bottleneck"]) is not int or model["protein_bottleneck"] <= 0:
        raise ValueError("protein_bottleneck must be a positive integer")
    if any(not math.isfinite(v) or v < 0 for v in loss.values()) or not any(loss.values()):
        raise ValueError("Loss weights must be finite, nonnegative and not all zero")
    for group, names in ((train, ("epochs", "batch_size", "patience")),
                         (evaluation, ("batch_size", "n_estimators", "min_samples_leaf")),
                         (extraction, ("batch_size", "max_length", "max_residues", "flush_every"))):
        if any(type(group[k]) is not int or group[k] <= 0 for k in names):
            raise ValueError(f"Expected positive integer options: {names}")
    if (train["num_workers"] < 0 or train["scheduler_patience"] < 0
            or not 0 <= train["seed"] < 2**32 or evaluation["n_jobs"] == 0):
        raise ValueError("Invalid seed/workers/patience/jobs")
    for key in ("lr", "adapter_lr", "new_head_lr"):
        if not math.isfinite(train[key]) or train[key] <= 0:
            raise ValueError(f"Invalid {key}")
    if not math.isfinite(train["weight_decay"]) or train["weight_decay"] < 0:
        raise ValueError("Invalid weight_decay")
    if not 0 < evaluation["max_features"] <= 1:
        raise ValueError("max_features is a fraction in (0, 1]")
    if not 0 <= extraction["overlap"] < extraction["max_residues"]:
        raise ValueError("Require 0 <= overlap < max_residues")
    if extraction["model_dtype"] not in ("float32", "float16", "bfloat16"):
        raise ValueError("Unsupported ESM-C dtype")


def load_config(path):
    path = Path(path).resolve()
    supplied = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(supplied, dict):
        raise ValueError(f"Configuration must be a JSON object: {path}")
    config = copy.deepcopy(DEFAULTS)
    for key, value in supplied.items():
        if key in DEFAULTS:
            if not isinstance(value, dict):
                raise ValueError(f"{path}: {key} must be a JSON object")
            config[key].update(value)
        else:
            config[key] = value
    try:
        validate(config)
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error
    def resolve(value):
        return str(project_path(value))
    config["data"] = {k: resolve(v) for k, v in config["data"].items()}
    config["output_dir"] = resolve(config["output_dir"])
    for key in ("random", "homology"):
        config["holdout"][key] = resolve(config["holdout"][key])
    output = Path(config["output_dir"])

    for key, value in config["data"].items():
        source = Path(value)
        if source.is_relative_to(output) or output.is_relative_to(source):
            raise ValueError(f"Output overlaps input {key}: {source}")
    return config
