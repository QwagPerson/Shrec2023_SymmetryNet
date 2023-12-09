import torch


def get_diagonals_length(points: torch.Tensor):
    """
    :param points: Shape N x 3
    :return: length Shape 1
    """
    diagonal = points.max(dim=0).values - points.min(dim=0).values
    return torch.linalg.norm(diagonal)
