import torch
from .alignment import ReactionAlignmentModel


def initialize_model(*, seed, protein_projection="residual", **options):


    torch.manual_seed(seed)
    model = ReactionAlignmentModel(protein_projection="residual", **options)
    if protein_projection == "linear":
        model.protein_projection = model.protein_projection.base
        model.architecture["protein_projection"] = "linear"
    elif protein_projection != "residual":
        raise ValueError(protein_projection)
    return model
