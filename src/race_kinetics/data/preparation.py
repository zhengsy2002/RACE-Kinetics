import hashlib
from functools import lru_cache
from pathlib import Path

import pandas as pd

from ..utils import sha256, write_json
from .caches import _normalize_sequence, _sequence_sha1
from .validation import reaction_key


def prepare_reactzyme(raw_path, protein_path, reaction_path, output):
    raw = pd.read_csv(raw_path, dtype=str).fillna("")
    proteins = pd.read_csv(protein_path, sep="\t", dtype=str).fillna("")
    reactions = pd.read_csv(reaction_path, sep="\t", dtype=str).fillna("")
    if not {"Entry", "Sequence", "Rhea ID"}.issubset(proteins):
        raise ValueError("proteins.tsv requires Entry, Sequence, Rhea ID")
    if not {"Rhea ID", "substrate", "product"}.issubset(reactions):
        raise ValueError("reactions.tsv requires Rhea ID, substrate, product")
    source_counts = {"source_protein_rows": len(proteins), "source_reaction_rows": len(reactions)}
    proteins["sequence_sha1"] = proteins.Sequence.map(_sequence_sha1)
    proteins["sequence_length"] = proteins.Sequence.map(_normalize_sequence).str.len()
    proteins = proteins[proteins.Entry.ne("") & proteins.sequence_length.gt(0)].drop_duplicates("Entry")
    key = lru_cache(maxsize=None)(reaction_key)
    reactions["key"] = [key(a, b) for a, b in zip(reactions.substrate, reactions["product"])]
    reactions = reactions[reactions["Rhea ID"].ne("") & reactions.key.notna()].drop_duplicates("Rhea ID")
    excluded_sequences = set(raw.sequence.map(_sequence_sha1))

    excluded_reactions = {key(a, b) for a, b in zip(raw.reactant_smiles, raw.product_smiles)} - {None}
    protein_hits = proteins.sequence_sha1.isin(excluded_sequences)
    reaction_hits = reactions.key.isin(excluded_reactions)
    stats = {**source_counts, "protocol": "kcat-only exact pretraining exclusion",
        "raw_sha256": sha256(Path(raw_path)), "kcat_rows": len(raw),
        "reaction_key": "canonical components; sorted within sides; direction invariant",
        "kcat_unique_sequence_exclusion_keys": len(excluded_sequences),
        "kcat_unique_complete_reaction_exclusion_keys": len(excluded_reactions),
        "kcat_rows_with_complete_products": int(raw.product_smiles.str.strip().ne("").sum()),
        "kcat_rows_missing_products_not_reaction_auditable": int(raw.product_smiles.str.strip().eq("").sum()),
        "reactzyme_protein_rows_hit_before_pair_pruning": int(protein_hits.sum()),
        "reactzyme_reaction_rows_hit": int(reaction_hits.sum()), "split_unit": "Rhea ID"}
    proteins = proteins[~protein_hits].reset_index(drop=True)
    reactions = reactions[~reaction_hits].reset_index(drop=True)
    rhea_rows = {value: i for i, value in enumerate(reactions["Rhea ID"])}
    records = []
    for row, (entry, ids) in enumerate(zip(proteins.Entry, proteins["Rhea ID"])):
        for rhea in dict.fromkeys(x.strip() for x in ids.split(";") if x.strip()):
            if rhea in rhea_rows:
                bucket = int(hashlib.sha1(rhea.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
                records.append((row, rhea_rows[rhea], entry, rhea, "valid" if bucket < .05 else "train"))
    pairs = pd.DataFrame(records, columns=["protein_row", "reaction_row", "entry", "rhea_id", "split"])
    pairs = pairs.drop_duplicates(["protein_row", "reaction_row"])
    if pairs.empty or set(pairs.split) != {"train", "valid"}:
        raise ValueError("Prepared pairs must contain nonempty train and valid splits")
    used = sorted(pairs.protein_row.unique())
    pairs["protein_row"] = pairs.protein_row.map({old: new for new, old in enumerate(used)})
    proteins = proteins.iloc[used].reset_index(drop=True)
    protein_index = proteins[["Entry", "sequence_sha1", "sequence_length"]].rename(columns={"Entry": "entry"})
    protein_index.insert(0, "protein_row", range(len(proteins)))
    reaction_index = reactions[["Rhea ID", "substrate", "product"]].rename(columns={"Rhea ID": "rhea_id"})
    reaction_index.insert(0, "reaction_row", range(len(reactions)))
    reaction_index["reaction_sha1"] = reactions.key.map(lambda s: hashlib.sha1(s.encode()).hexdigest())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    from ..runtime import atomic_output
    for name, table in (("protein_index.tsv", protein_index), ("reaction_index.tsv", reaction_index), ("pairs.tsv", pairs)):
        with atomic_output(output / name) as temp:
            table.to_csv(temp, sep="\t", index=False, lineterminator="\n")
    stats.update(proteins=len(proteins), reactions=len(reactions), pairs=len(pairs),
                 train_pairs=int(pairs.split.eq("train").sum()), valid_pairs=int(pairs.split.eq("valid").sum()))
    write_json(output / "prepare_stats.json", stats)
    return stats

