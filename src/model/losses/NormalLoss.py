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

        return REDUCTIONS[self.reduction](1 - torch.abs(torch.linalg.vecdot(n_true, n_pred)))
