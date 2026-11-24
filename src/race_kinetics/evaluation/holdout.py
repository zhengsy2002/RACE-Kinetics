from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

from .metrics import compute_metrics
from ..utils import sha256, write_json
from ..storage import freeze_record, read_record, write_record, atomic_csv

METRICS = ("pcc", "scc", "r2", "rmse", "mae", "p1mag")


def make_regressor(settings, seed):
    return ExtraTreesRegressor(random_state=seed, **{k: settings[k] for k in
        ("n_estimators", "min_samples_leaf", "max_features", "n_jobs")})


def predictions(frame, indices, values, protocol, seed):
    return pd.DataFrame({"sample_id": frame.sample_id.iloc[indices].to_numpy(),
                         "row_index": indices, "y_true": frame.log10_value.iloc[indices].to_numpy(),
                         "y_pred": values, "protocol": protocol, "seed": seed})


def completed(output, stage):
    record = read_record(Path(output) / f"{stage}_complete.json")
    if record is None:
        weights = Path(output) / "weights/regressor.joblib"
        if stage == "fit" and weights.is_file():
            return getattr(joblib.load(weights), "race_kinetics_provenance_", None) is not None
        return False
    for name, digest in record.items():
        p = Path(output) / name
        if not p.is_file() or sha256(p) != digest:
            raise ValueError(f"Completed {stage} artifact changed: {p}")
    return True


def mark(output, stage, files):
    write_record(Path(output) / f"{stage}_complete.json",
                 {p.relative_to(output).as_posix(): sha256(p) for p in files})


def cross_validate(frame, features, assignments, settings, seed, output, protocol):
    output = Path(output)
    if completed(output, "validation"):
        return
    rows = []
    tables = []
    for fold in range(10):
        train = np.flatnonzero((assignments >= 0) & (assignments != fold))
        valid = np.flatnonzero(assignments == fold)
        if not len(train) or not len(valid):
            raise ValueError("Empty inner training or validation fold")
        model = make_regressor(settings, seed + fold)
        model.fit(features[train], frame.log10_value.iloc[train].to_numpy())
        values = model.predict(features[valid])
        tables.append(predictions(frame, valid, values, protocol, seed).assign(inner_fold=fold))
        rows.append({"inner_fold": fold, "train_rows": len(train), "valid_rows": len(valid),
                     **compute_metrics(frame.log10_value.iloc[valid].to_numpy(), values)})
        print(f"{protocol} seed={seed} inner_fold={fold}: train={len(train)} valid={len(valid)}", flush=True)
    table = pd.concat(tables).sort_values("row_index")
    if len(table) != int((assignments >= 0).sum()) or not table.sample_id.is_unique:
        raise ValueError("Incomplete development OOF predictions")
    paths = [output / "predictions/development_oof.csv", output / "evaluation/inner_fold_metrics.csv",
             output / "evaluation/development.json"]
    atomic_csv(table, paths[0])
    metrics = pd.DataFrame(rows)
    atomic_csv(metrics, paths[1])
    write_json(paths[2], {"scope": "development_only", "oof": compute_metrics(table.y_true.to_numpy(), table.y_pred.to_numpy()),
                         "fold_mean": {m: float(metrics[m].mean()) for m in METRICS},
                         "fold_std": {m: float(metrics[m].std(ddof=1)) for m in METRICS}, "ddof": 1})
    mark(output, "validation", paths)


def fit_final(frame, features, assignments, settings, seed, output, identity):
    output = Path(output)
    development = np.flatnonzero(assignments >= 0)
    path = output / "weights/regressor.joblib"
    if path.is_file():
        saved = getattr(joblib.load(path), "race_kinetics_provenance_", None)
        if saved is None or saved["identity"] != identity or saved["training_sample_ids"] != frame.sample_id.iloc[development].tolist():
            raise ValueError("Existing regressor does not match these inputs/configuration; use a new results directory")
    if completed(output, "fit"):
        return
    model = make_regressor(settings, seed)
    model.fit(features[development], frame.log10_value.iloc[development].to_numpy())
    model.race_kinetics_provenance_ = {"identity": identity,
        "training_sample_ids": frame.sample_id.iloc[development].tolist()}
    path.parent.mkdir(parents=True, exist_ok=True)
    from ..runtime import atomic_output
    with atomic_output(path) as temp:
        joblib.dump(model, temp, compress=3)
    mark(output, "fit", [path])


def test_final(frame, features, assignments, seed, output, protocol, identity):
    output = Path(output)
    if not completed(output, "fit"):
        raise FileNotFoundError("Run --stage fit before --stage test")
    model = joblib.load(output / "weights/regressor.joblib")
    provenance = getattr(model, "race_kinetics_provenance_", None)
    development = np.flatnonzero(assignments >= 0)
    if (provenance is None or provenance["identity"] != identity or
            provenance["training_sample_ids"] != frame.sample_id.iloc[development].tolist()):
        raise ValueError("Final model was not fitted on the specified development set")
    if completed(output, "test"):
        return
    test = np.flatnonzero(assignments == -1)
    values = model.predict(features[test])
    table = predictions(frame, test, values, protocol, seed)
    paths = [output / "predictions/test.csv", output / "evaluation/test.json"]
    atomic_csv(table, paths[0])
    write_json(paths[1], {"scope": "fixed_test", "protocol": protocol, "seed": seed,
                          "test_rows": len(test), "metrics": compute_metrics(table.y_true.to_numpy(), values)})
    mark(output, "test", paths)


def run_evaluation(config, frame, features, assignments, split_path, feature_path, stage, seed, protocol, output):
    from ..utils import file_identity
    output = Path(output)
    identity = file_identity([Path(config["data"]["raw_csv"]), Path(split_path), Path(feature_path)],
        {"protocol": protocol, "seed": seed, "evaluation": config["evaluation"], "features": config["features"]})
    weights = output / "weights/regressor.joblib"
    if weights.is_file():
        saved = getattr(joblib.load(weights), "race_kinetics_provenance_", None)
        if saved is None or saved["identity"] != identity:
            raise ValueError("Existing regressor inputs/configuration changed; use a new results directory")
    freeze_record(output / "identity.json", identity)
    if stage in ("validate", "all"):
        cross_validate(frame, features, assignments, config["evaluation"], seed, output, protocol)
    if stage in ("fit", "all"):
        fit_final(frame, features, assignments, config["evaluation"], seed, output, identity)
    if stage in ("test", "all"):
        test_final(frame, features, assignments, seed, output, protocol, identity)


def summarize(root, protocols, experiments, seeds):
    import json
    rows = []
    for protocol in protocols:
        for experiment in experiments:
            for seed in seeds:
                output = Path(root) / protocol / experiment / f"seed_{seed}"
                if completed(output, "test"):
                    value = json.loads((output / "evaluation/test.json").read_text())
                    rows.append({"protocol": protocol, "experiment": experiment, "seed": seed, **value["metrics"]})
    if not rows:
        return
    table = pd.DataFrame(rows)
    atomic_csv(table, Path(root) / "evaluation/test_metrics.csv")
    records = []
    for (protocol, experiment), group in table.groupby(["protocol", "experiment"]):
        complete = set(group.seed) == set(seeds)
        records.append({"protocol": protocol, "experiment": experiment, "completed_seeds": len(group),
                        "expected_seeds": len(seeds), "complete": complete,
                        **{f"{m}_mean": float(group[m].mean()) if complete else None for m in METRICS},
                        **{f"{m}_std": float(group[m].std(ddof=1)) if complete and len(group) > 1 else None for m in METRICS}})
    atomic_csv(pd.DataFrame(records), Path(root) / "evaluation/test_summary.csv")
