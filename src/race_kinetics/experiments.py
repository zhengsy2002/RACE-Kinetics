import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

from .config import PROJECT_ROOT, load_config, project_path, configure_runtime, validate_paths
from .utils import sha256, bind_preparation_sources
from .storage import bind_record, checkpoint_file, runtime_session

EXPERIMENTS = ("full", "no_er", "no_rp", "no_dir", "no_environment", "learned_only")


def arm_config(base, name):
    config = copy.deepcopy(base)
    if name in ("no_er", "no_rp", "no_dir"):
        config["loss"][{"no_er": "enzyme_reaction", "no_rp": "reactant_product", "no_dir": "direction"}[name]] = 0.0
    if name == "no_environment":
        config["features"]["drop_assay"] = ["ph", "temperature"]
    if name == "learned_only":
        config["features"]["mode"] = "learned_only"
    return config


def audit_splits(config, protocols):
    from .data.kcat import native_frame
    from .data.holdout import read_partition
    frame = native_frame(config["data"]["raw_csv"])
    splits = {}
    reports = {}
    for protocol in protocols:
        path = Path(config["holdout"][protocol])
        table, assignments, report = read_partition(frame, path, protocol)
        report["dataset_sha256"] = sha256(Path(config["data"]["raw_csv"]))
        reports[protocol] = report
        splits[protocol] = (table, assignments, path)
    return frame, splits, reports


def prepare(config):
    from .data.kcat import prepare_kcat
    from .data.preparation import prepare_reactzyme
    from .data.validation import audit_data
    data = config["data"]
    sources = [Path(data[k]) for k in ("raw_csv", "protein_tsv", "reaction_tsv")]
    missing = [str(p) for p in sources if not p.is_file()]
    if missing:
        raise FileNotFoundError("Required input files missing: " + ", ".join(missing))
    prepare_kcat(data)
    folder = Path(data["prepared_dir"])
    folder.mkdir(parents=True, exist_ok=True)
    bind_preparation_sources(folder / "source_binding.json", sources,
                             {"exclusion": "sequence OR complete undirected reaction"})
    if not (folder / "prepare_stats.json").is_file():
        prepare_reactzyme(*sources, folder)
    return audit_data(data)


def checkpoint_for(config, name, stage):
    from .pipeline import training_args
    from .features.alignment import _assert_pretraining_complete, _load_checkpoint
    arm = "full" if name in ("no_environment", "learned_only") else name
    pretrain_config = arm_config(config, arm)
    output = Path(config["output_dir"]) / "pretraining" / arm
    pretrain_config["output_dir"] = str(output)
    files = [Path(config["data"]["prepared_dir"]) / n for n in
             ("protein_index.tsv", "reaction_index.tsv", "pairs.tsv")]
    files += [Path(config["data"][key]) / n for key, names in {
        "protein_pretrain_cache": ["unique_embeddings.npy", "protein_row_to_cache_row.npy", "unique_sequences.csv"],
        "molformer_cache": ["side_embeddings.npy", "unique_sides.csv"]}.items() for n in names]
    checkpoint = checkpoint_file(output)
    existing_binding = None
    if checkpoint.exists():
        existing_binding = _load_checkpoint(checkpoint).get("pretraining_binding")
        if existing_binding is None:
            raise ValueError("Checkpoint lacks embedded training-input binding")
    binding = bind_record(output / "pretraining_binding.json", files,
        {k: pretrain_config[k] for k in ("model", "loss", "training", "supervision")})
    if existing_binding is not None and existing_binding != binding:
        raise ValueError("Pretrained model inputs/configuration/code changed; use a new results directory")
    if stage in ("all", "pretrain"):
        from .training.trainer import train
        done = False
        if checkpoint.is_file():
            try:
                _assert_pretraining_complete(checkpoint)
                done = True
            except RuntimeError:
                pass
        if not done:
            train(training_args(pretrain_config))
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Run --stage pretrain first: {checkpoint}")
    _assert_pretraining_complete(checkpoint)
    state = _load_checkpoint(checkpoint)
    if state.get("loss_weights") != pretrain_config["loss"] or state.get("supervision") != pretrain_config["supervision"]:
        raise ValueError("Pretrained checkpoint objective settings differ")
    return checkpoint


def run(config, names, protocols, stage):
    from .data.validation import audit_data, audit_proteins, audit_molecules
    from .data.kcat import prepare_kcat
    from .pipeline import extract_features
    from .features.matrix import build_matrix
    from .evaluation.holdout import run_evaluation, summarize
    validate_paths(config)
    _, splits, reports = audit_splits(config, protocols)
    print(json.dumps(reports, indent=2), flush=True)
    if stage == "check":
        from .backbones.download import model_status
        missing = {k: config["data"][k] for k in ("protein_tsv", "reaction_tsv")
                   if not Path(config["data"][k]).is_file()}
        for key in ("esmc_model", "molformer_model"):
            folder = Path(config["data"][key])
            unavailable = model_status(folder, key.removesuffix("_model"))
            if unavailable:
                missing[key] = {"folder": str(folder), "missing_or_invalid": unavailable,
                                "auto_download_on_extract": config["downloads"]["auto_download"]}
        print(json.dumps({"missing_resources": missing, "source_files_available": not missing,
                          "note": "Cache identity and target exclusion are checked before training."}, indent=2))
        return
    temporary = configure_runtime()
    root = Path(config["output_dir"])
    lock_key = hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:16]
    lock = temporary.parent / f"training-{lock_key}.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"Another run or interrupted process owns {lock}. Check the recorded PID before removing it.") from exc
    with os.fdopen(descriptor, "w") as stream:
        stream.write(str(os.getpid()))
    try:
        if stage in ("prepare", "all"):
            frame, proteins, reactions, report = prepare(config)
        else:
            prepare_kcat(config["data"])
            frame, proteins, reactions, report = audit_data(config["data"])
        print("Pretraining exclusion check:", json.dumps(report), flush=True)
        if stage == "prepare":
            return
        if stage in ("extract", "all"):
            extract_features(config)
        audit_proteins(config["data"], frame, proteins)
        audit_molecules(config["data"], frame, reactions)
        if stage == "extract":
            return
        for name in names:
            current = arm_config(config, name)
            checkpoint = checkpoint_for(config, name, stage)
            if stage == "pretrain":
                continue
            matrix, matrix_path = build_matrix(current, frame, checkpoint)
            if stage == "features":
                continue
            for protocol in protocols:
                _, assignments, split_path = splits[protocol]
                for seed in config["holdout"]["seeds"]:
                    output = root / protocol / name / f"seed_{seed}"
                    run_evaluation(current, frame, matrix, assignments, split_path, matrix_path,
                                   stage, seed, protocol, output)
        if stage in ("test", "all"):
            summarize(root, protocols, names, config["holdout"]["seeds"])
    finally:
        lock.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train and evaluate RACE-Kinetics with fixed holdout partitions")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/default.json"))
    parser.add_argument("--protocol", choices=("random", "homology", "both"), default="both")
    parser.add_argument("--experiments", default="full", help="Comma-separated names or all")
    parser.add_argument("--stage", default="check", choices=("check", "prepare", "extract",
        "pretrain", "features", "validate", "fit", "test", "all"))
    parser.add_argument("--seeds", help="Comma-separated downstream regressor seeds")
    parser.add_argument("--results-dir")
    args = parser.parse_args(argv)
    config = load_config(project_path(args.config))
    if args.seeds:
        config["holdout"]["seeds"] = [int(s) for s in args.seeds.split(",")]
    if args.results_dir:
        config["output_dir"] = str(project_path(args.results_dir))
    from .config import validate
    validate(config)
    names = list(EXPERIMENTS) if args.experiments == "all" else args.experiments.split(",")
    if len(set(names)) != len(names) or set(names) - set(EXPERIMENTS):
        parser.error(f"Experiments must be unique members of {EXPERIMENTS}")
    protocols = ["random", "homology"] if args.protocol == "both" else [args.protocol]
    with runtime_session():
        run(config, names, protocols, args.stage)
