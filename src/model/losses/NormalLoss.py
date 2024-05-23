import warnings

import torch
from torch import nn

REDUCTIONS = {
    "mean": torch.mean,
    "sum": torch.sum,
}


class NormalLoss(nn.Module):
    def __init__(self, check_normalized=True, reduction="mean"):
        super().__init__()
        self.check_normalized = check_normalized
        self.reduction = reduction

    def forward(self, n_pred, n_true):
        if self.check_normalized:
            pred_normalized = torch.allclose(torch.norm(n_pred, dim=1),
                                             torch.ones((n_pred.shape[0]), device=n_pred.device))
            true_normalized = torch.allclose(torch.norm(n_true, dim=1),
                                             torch.ones((n_true.shape[0]), device=n_true.device))
            if not pred_normalized:
                warnings.warn("Got n_pred with normals that are not normalized!")
            if not true_normalized:
                warnings.warn("Got n_true with normals that are not normalized!")

        # 1 - |a.b/(|a||b|))| < 0.00015230484 is good
        # |a.b/(|a||b|) = cos(angle between a and b)
        abs_angle = torch.abs(
            torch.linalg.vecdot(n_true, n_pred) / (torch.norm(n_true, dim=1) * torch.norm(n_pred, dim=1))
        )

        return REDUCTIONS[self.reduction](1 - abs_angle)
