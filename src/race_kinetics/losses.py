from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def contrastive_loss(left, right, scale, positives):
    logits = scale * left @ right.t()
    negative_inf = torch.finfo(logits.dtype).min
    row_num = torch.logsumexp(logits.masked_fill(~positives, negative_inf), dim=1)
    row_den = torch.logsumexp(logits, dim=1)
    col_num = torch.logsumexp(logits.t().masked_fill(~positives.t(), negative_inf), dim=1)
    col_den = torch.logsumexp(logits.t(), dim=1)
    loss = -0.5 * ((row_num - row_den).mean() + (col_num - col_den).mean())
    top = logits.argmax(dim=1)
    recall = positives[torch.arange(len(top), device=top.device), top].float().mean().item()
    return loss, recall


def direction_loss(forward_logit, reverse_logit, eligible=None):
    if eligible is not None:
        if not eligible.any():
            return (forward_logit.sum() + reverse_logit.sum()) * 0, 0.0
        forward_logit, reverse_logit = forward_logit[eligible], reverse_logit[eligible]
    logits = torch.cat([forward_logit, reverse_logit])
    targets = torch.cat([torch.ones_like(forward_logit), torch.zeros_like(reverse_logit)])
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    accuracy = ((logits >= 0) == targets.bool()).float().mean().item()
    return loss, accuracy


class AlignmentLoss(nn.Module):
    def __init__(self, enzyme_reaction=1.0, reactant_product=1.0, direction=1.0):
        super().__init__()
        self.weights = dict(enzyme_reaction=enzyme_reaction,
                            reactant_product=reactant_product, direction=direction)
        if any(not math.isfinite(v) or v < 0 for v in self.weights.values()) or not any(self.weights.values()):
            raise ValueError("Loss weights must be finite, nonnegative and not all zero")

    def forward(self, output, batch):
        protein_rows, reaction_rows = batch["protein_row"], batch["reaction_row"]
        rp_mask = reaction_rows[:, None] == reaction_rows[None, :]
        er_mask = rp_mask | (protein_rows[:, None] == protein_rows[None, :])
        er, er_recall = contrastive_loss(output["protein"], output["reaction"],
                                         output["enzyme_reaction_scale"], er_mask)
        rp, rp_recall = contrastive_loss(output["reactant"], output["product"],
                                         output["reactant_product_scale"], rp_mask)
        direction, accuracy = direction_loss(output["forward_direction_logit"],
                                             output["reverse_direction_logit"],
                                             batch.get("direction_eligible"))
        terms = dict(enzyme_reaction=er, reactant_product=rp, direction=direction)
        total = sum(self.weights[key] * loss for key, loss in terms.items() if self.weights[key] != 0)
        return total, {
            "loss": total.detach().item(),
            "enzyme_reaction_loss": er.detach().item(),
            "enzyme_reaction_recall_at_1": er_recall,
            "reactant_product_loss": rp.detach().item(),
            "reactant_product_recall_at_1": rp_recall,
            "direction_loss": direction.detach().item(),
            "direction_accuracy": accuracy,
        }
