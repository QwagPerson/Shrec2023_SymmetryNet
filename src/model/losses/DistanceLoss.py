import torch
from torch import nn

REDUCTIONS = {
    "mean": torch.mean,
    "sum": torch.sum,
}


class DistanceLoss(nn.Module):
    def __init__(self, p=2, reduction="mean"):
        super().__init__()
        self.p = p
        self.reduction = reduction

    def forward(self, center_pred, center_true):
        distances = torch.norm(center_true - center_pred, p=self.p, dim=0)
        return REDUCTIONS[self.reduction](distances)
