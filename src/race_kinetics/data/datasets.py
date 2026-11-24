from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
class MolFormerPairDataset(Dataset):
    def __init__(
        self,
        pairs: pd.DataFrame,
        protein_cache: np.ndarray,
        protein_row_to_cache_row: np.ndarray,
        reactant_rows: np.ndarray,
        product_rows: np.ndarray,
        side_embeddings: np.ndarray,
    ) -> None:
        self.pairs = pairs.reset_index(drop=True)
        self.protein_cache = protein_cache
        self.protein_row_to_cache_row = protein_row_to_cache_row
        self.reactant_rows = reactant_rows
        self.product_rows = product_rows
        self.side_embeddings = side_embeddings

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, object]:
        pair = self.pairs.iloc[index]
        protein_row = int(pair["protein_row"])
        reaction_row = int(pair["reaction_row"])
        return {
            "protein": np.asarray(
                self.protein_cache[int(self.protein_row_to_cache_row[protein_row])]
            ).copy(),
            "reactant_molformer": np.asarray(
                self.side_embeddings[int(self.reactant_rows[reaction_row])]
            ).copy(),
            "product_molformer": np.asarray(
                self.side_embeddings[int(self.product_rows[reaction_row])]
            ).copy(),
            "protein_row": protein_row,
            "reaction_row": reaction_row,
        }


def collate(items: list[dict[str, object]]) -> dict[str, torch.Tensor]:
    return {
        "protein": torch.from_numpy(np.stack([item["protein"] for item in items])).float(),
        "reactant_molformer": torch.from_numpy(
            np.stack([item["reactant_molformer"] for item in items])
        ).float(),
        "product_molformer": torch.from_numpy(
            np.stack([item["product_molformer"] for item in items])
        ).float(),
        "protein_row": torch.tensor([item["protein_row"] for item in items], dtype=torch.long),
        "reaction_row": torch.tensor([item["reaction_row"] for item in items], dtype=torch.long),
    }
