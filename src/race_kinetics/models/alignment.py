from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from .projections import make_protein_projection


def projection_block(dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(256),
        nn.Linear(256, 256),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(256, 256),
    )


class MolFormerAdapter(nn.Module):
    def __init__(self, input_dim: int = 768, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.network = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class ReactionAlignmentModel(nn.Module):


    def __init__(self, *, molformer_dim: int = 768, dropout: float = 0.1,
                 protein_input_dim: int = 1152, protein_projection: str = "linear",
                 protein_bottleneck: int = 128) -> None:
        super().__init__()
        self.molformer_dim = int(molformer_dim)
        self.protein_input_dim = int(protein_input_dim)
        self.architecture = {
            "protein_projection": protein_projection,
            "protein_bottleneck": protein_bottleneck,
            "dropout": dropout,
        }
        self.protein_projection = make_protein_projection(
            protein_projection, self.protein_input_dim, protein_bottleneck, dropout
        )
        self.molecule_adapter = MolFormerAdapter(self.molformer_dim, dropout)
        self.reaction_projection = nn.Sequential(
            nn.LayerNorm(256 * 3),
            nn.Linear(256 * 3, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
        )
        self.reactant_projection = projection_block(dropout)
        self.product_projection = projection_block(dropout)
        self.direction_head = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )
        initial_scale = math.log(1.0 / 0.07)
        self.enzyme_reaction_logit_scale = nn.Parameter(torch.tensor(initial_scale))
        self.reactant_product_logit_scale = nn.Parameter(torch.tensor(initial_scale))

    @staticmethod
    def _scale(parameter: torch.Tensor) -> torch.Tensor:
        return parameter.exp().clamp(max=100.0)

    def encode_molecule(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.molecule_adapter(embedding)

    def direction_logits(self, forward_features, reverse_features):
        return (self.direction_head(forward_features).squeeze(-1),
                self.direction_head(reverse_features).squeeze(-1))

    def forward(self, batch: dict[str, object]) -> dict[str, torch.Tensor]:
        protein_raw = self.protein_projection(batch["protein"])
        reactant_raw = self.encode_molecule(batch["reactant_molformer"])
        product_raw = self.encode_molecule(batch["product_molformer"])
        forward_features = torch.cat(
            [reactant_raw, product_raw, product_raw - reactant_raw], dim=-1
        )
        reverse_features = torch.cat(
            [product_raw, reactant_raw, reactant_raw - product_raw], dim=-1
        )
        reaction_raw = self.reaction_projection(forward_features)
        reactant = self.reactant_projection(reactant_raw)
        product = self.product_projection(product_raw)
        forward_logit, reverse_logit = self.direction_logits(
            forward_features, reverse_features)
        return {
            "protein": F.normalize(protein_raw, dim=-1),
            "reaction": F.normalize(reaction_raw, dim=-1),
            "reactant": F.normalize(reactant, dim=-1),
            "product": F.normalize(product, dim=-1),
            "forward_direction_logit": forward_logit,
            "reverse_direction_logit": reverse_logit,
            "enzyme_reaction_scale": self._scale(self.enzyme_reaction_logit_scale),
            "reactant_product_scale": self._scale(self.reactant_product_logit_scale),
        }


def checkpoint_components(model: ReactionAlignmentModel) -> dict[str, object]:
    return {
        "protein_input_dim": model.protein_input_dim,
        "architecture": dict(model.architecture),
        "protein_projection": model.protein_projection.state_dict(),
        "molecule_adapter": model.molecule_adapter.state_dict(),
        "reaction_projection": model.reaction_projection.state_dict(),
        "reactant_projection": model.reactant_projection.state_dict(),
        "product_projection": model.product_projection.state_dict(),
        "direction_head": model.direction_head.state_dict(),
        "enzyme_reaction_logit_scale": model.enzyme_reaction_logit_scale.detach().cpu(),
        "reactant_product_logit_scale": model.reactant_product_logit_scale.detach().cpu(),
    }
