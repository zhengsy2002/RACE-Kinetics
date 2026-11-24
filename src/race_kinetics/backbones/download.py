import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from ..config import PROJECT_ROOT, configure_runtime, project_path, validate_paths


def model_status(folder, name):
    folder = project_path(folder)
    for path in folder.rglob("*"):
        if path.is_file() or path.is_symlink():
            project_path(path)
    missing = []
    documents = {}
    for filename in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        path = folder / filename
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(document, dict):
                raise ValueError(f"Expected a JSON object in {filename}")
            documents[filename] = document
        except (OSError, ValueError):
            missing.append(filename)
    config = documents.get("config.json")
    if config is not None:
        expected = ("esmc", 1152) if name == "esmc" else ("molformer", 768)
        if (config.get("model_type"), config.get("hidden_size")) != expected:
            raise ValueError(f"Wrong {name} encoder at {folder}; expected type/dimension {expected}")
        for reference in config.get("auto_map", {}).values():
            for item in reference if isinstance(reference, list) else [reference]:
                if item:
                    if "--" in item:
                        raise ValueError(f"Remote code reference is not local to this encoder: {item}")
                    filename = item.rsplit(".", 1)[0] + ".py"
                    module = project_path(folder / filename)
                    if not module.is_file() or not module.stat().st_size:
                        missing.append(filename)
    weights = [folder / "model.safetensors"]
    index = folder / "model.safetensors.index.json"
    if index.is_file():
        try:
            shards = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
            weights = [project_path(folder / filename) for filename in sorted(set(shards.values()))]
            if not weights:
                missing.append(index.name)
        except (OSError, KeyError, TypeError, ValueError):
            missing.append(index.name)
    for weight in weights:
        if not weight.is_file() or not weight.stat().st_size:
            missing.append(weight.name)
            continue
        from safetensors import safe_open
        try:
            with safe_open(str(weight), framework="np") as tensors:
                if not list(tensors.keys()):
                    missing.append(weight.name)
        except Exception:
            missing.append(weight.name)
    return sorted(set(missing))


def _download(repo_id, revision, folder):
    scratch = Path(tempfile.mkdtemp(prefix="download-", dir=configure_runtime()))
    environment = os.environ.copy()
    environment.update(HF_HUB_OFFLINE="0", TRANSFORMERS_OFFLINE="0",
                       PYTHONPATH=str(PROJECT_ROOT / "src"))
    command = [sys.executable, "-m", "race_kinetics.backbones.download",
               "--repo-id", repo_id, "--revision", revision, "--folder", str(folder),
               "--scratch", str(scratch)]
    try:
        subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"Encoder download failed: {repo_id}@{revision} -> {folder}. "
            "Check network access to huggingface.co (or HF_ENDPOINT), then rerun the same command. "
            "If the repository requires access, set HF_TOKEN. No encoder from another project is used."
        ) from error


def ensure_models(config, names=("esmc", "molformer")):
    validate_paths(config)
    options = config["downloads"]
    for name in names:
        if name not in ("esmc", "molformer"):
            raise ValueError(f"Unknown encoder: {name}")
        folder = project_path(config["data"][f"{name}_model"])
        missing = model_status(folder, name)
        if not missing:
            print(f"Use complete local {name} encoder: {folder}", flush=True)
            continue
        if not options["auto_download"]:
            raise FileNotFoundError(f"Incomplete {name} encoder in {folder}: {missing}; enable downloads.auto_download")
        print(f"Download {name}: {options[f'{name}_repo_id']} -> {folder}; missing/invalid: {missing}", flush=True)
        folder.mkdir(parents=True, exist_ok=True)
        _download(options[f"{name}_repo_id"], options[f"{name}_revision"], folder)
        missing = model_status(folder, name)
        if missing:
            raise RuntimeError(f"Downloaded {name} encoder is incomplete: {missing}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--folder", required=True)
    parser.add_argument("--scratch", required=True)
    args = parser.parse_args()
    folder = project_path(args.folder)
    scratch = project_path(args.scratch)
    if not scratch.is_relative_to(project_path("data/tmp")) or not scratch.is_dir():
        raise ValueError("Encoder download scratch directory must belong to the active project runtime")
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=args.repo_id, revision=args.revision, local_dir=scratch,
                      cache_dir=Path(os.environ["HF_HUB_CACHE"]),
                      allow_patterns=["*.json", "*.safetensors", "*.py", "*.txt", "*.model", "README.md"],
                      max_workers=4, local_files_only=False)
    for source in sorted(scratch.rglob("*")):
        relative = source.relative_to(scratch)
        if source.is_file() and not any(part.startswith(".") for part in relative.parts):
            destination = project_path(folder / relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.replace(destination)


if __name__ == "__main__":
    main()
