from __future__ import annotations

import torch
from torch import nn


class ResidualProteinProjection(nn.Module):

    def __init__(self, input_dim=1152, output_dim=256, bottleneck=128, dropout=0.1):
        super().__init__()
        self.base = nn.Linear(input_dim, output_dim)
        self.residual = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.Linear(output_dim, bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, output_dim),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        projected = self.base(embedding)
        return projected + self.residual(projected)


def make_protein_projection(kind, input_dim, bottleneck=128, dropout=0.1):
    if kind == "linear":
        return nn.Linear(input_dim, 256)
    if kind == "residual":
        if type(bottleneck) is not int or bottleneck <= 0:
            raise ValueError("protein_bottleneck must be a positive integer")
        return ResidualProteinProjection(input_dim, 256, bottleneck, dropout)
    raise ValueError(f"Unsupported protein projection: {kind}")


def checkpoint_model_options(checkpoint):
    architecture = checkpoint.get("architecture")
    if not isinstance(architecture, dict) or set(architecture) != {
            "protein_projection", "protein_bottleneck", "dropout"}:
        raise ValueError("Checkpoint requires explicit current architecture metadata")
    if any(k not in checkpoint for k in ("molecule_adapter", "reactant_projection", "product_projection")):
        raise ValueError("Shared molecular adapter or RP projection heads missing in checkpoint")
    kind = architecture["protein_projection"]
    if kind not in ("linear", "residual"):
        raise ValueError(f"Unsupported checkpoint protein projection: {kind}")
    state = checkpoint["protein_projection"]
    key = "weight" if kind == "linear" else "base.weight"
    if key not in state or state[key].ndim != 2 or state[key].shape[0] != 256:
        raise ValueError("Checkpoint protein projection does not match its architecture")
    input_dim = int(state[key].shape[1])
    if checkpoint.get("protein_input_dim") != input_dim:
        raise ValueError("Checkpoint protein_input_dim differs from projection weights")
    bottleneck = architecture["protein_bottleneck"]
    if type(bottleneck) is not int or bottleneck <= 0:
        raise ValueError("Invalid checkpoint protein_bottleneck")
    if kind == "residual":
        if ("residual.1.weight" not in state or "residual.4.weight" not in state
                or tuple(state["residual.1.weight"].shape) != (bottleneck, 256)
                or tuple(state["residual.4.weight"].shape) != (256, bottleneck)):
            raise ValueError("Checkpoint residual bottleneck dimensions do not match metadata")
    head = checkpoint["direction_head"]
    if "1.weight" not in head or tuple(head["1.weight"].shape) != (256, 768):
        raise ValueError("Checkpoint DIR head dimensions do not match the model")
    result = {
        "protein_input_dim": input_dim,
        "molformer_dim": int(checkpoint["molformer_dim"]),
        "protein_projection": kind,
        "protein_bottleneck": bottleneck,
        "dropout": float(architecture["dropout"]),
    }
    return result
