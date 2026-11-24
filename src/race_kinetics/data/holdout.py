import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils import sha256


def read_partition(frame, path, protocol):
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"sample_id", "row_index", "partition", "inner_fold", "observation_group"}
    if protocol == "homology":
        required |= {"component", "sequence_sha256"}
    if not required.issubset(table):
        raise ValueError(f"Split columns missing: {sorted(required - set(table))}")
    ids = frame.sample_id.astype(str)
    if not ids.is_unique or ids.str.strip().eq("").any():
        raise ValueError("Dataset sample IDs must be unique and nonempty")
    if (len(table) != len(frame) or not table.sample_id.is_unique
            or set(table.sample_id) != set(ids)):
        raise ValueError("Split must cover every dataset sample exactly once")
    table = table.set_index("sample_id").loc[ids].reset_index()
    row_indices = pd.to_numeric(table.row_index, errors="raise").to_numpy()
    if not np.array_equal(row_indices, np.arange(len(frame))):
        raise ValueError("Split row_index does not match dataset sample IDs/order")
    if set(table.partition) != {"development", "test"}:
        raise ValueError("Partitions must be development and test")
    development = table.partition.eq("development").to_numpy()
    if table.loc[~development, "inner_fold"].ne("").any():
        raise ValueError("Test records must not have an inner fold")
    fold = pd.to_numeric(table.loc[development, "inner_fold"], errors="raise").to_numpy()
    if not np.isfinite(fold).all() or not np.array_equal(fold, fold.astype(int)) or set(fold) != set(range(10)):
        raise ValueError("Development folds must contain integer labels 0..9")
    assignments = np.full(len(frame), -1, dtype=int)
    assignments[development] = fold.astype(int)
    for key in ("observation_group", "component") if protocol == "homology" else ("observation_group",):
        if table[key].str.strip().eq("").any():
            raise ValueError(f"Empty {key}")
        if (table.groupby(key).partition.nunique() > 1).any():
            raise ValueError(f"{key} crosses development/test")
        if (table.loc[development].groupby(key).inner_fold.nunique() > 1).any():
            raise ValueError(f"{key} crosses inner folds")
    sequence = frame.sequence.astype(str).str.replace(r"\s+", "", regex=True).str.upper()
    if protocol == "homology":
        hashes = sequence.map(lambda s: hashlib.sha256(s.encode()).hexdigest())
        if not np.array_equal(hashes, table.sequence_sha256):
            raise ValueError("Split sequence hashes do not match current dataset")
        if (table.assign(sequence=sequence.to_numpy()).groupby("sequence").component.nunique() > 1).any():
            raise ValueError("Identical sequences have different homology components")
    overlap = set(sequence[development]) & set(sequence[~development])
    if protocol == "homology" and overlap:
        raise ValueError("Identical sequences cross homology holdout")
    report = {"protocol": protocol, "split_sha256": sha256(Path(path)), "rows": len(frame),
              "development_rows": int(development.sum()), "test_rows": int((~development).sum()),
              "inner_fold_counts": {str(i): int((assignments == i).sum()) for i in range(10)},
              "exact_sequence_overlap": len(overlap),
              "homology_claim": "Validates supplied component membership; does not rerun sequence search"}
    return table, assignments, report
