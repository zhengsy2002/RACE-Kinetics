from __future__ import annotations
import pandas as pd
def _reaction_roles(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    substrates = frame["reactant_smiles"].fillna("").astype(str).tolist()
    reactants = frame["reactant_smiles"].fillna("").astype(str).tolist()
    products = frame["product_smiles"].fillna("").astype(str).tolist()
    reactants = [
        value if value.strip() else substrate
        for value, substrate in zip(reactants, substrates)
    ]
    products = [
        value if value.strip() else reactant
        for value, reactant in zip(products, reactants)
    ]
    return reactants, products

