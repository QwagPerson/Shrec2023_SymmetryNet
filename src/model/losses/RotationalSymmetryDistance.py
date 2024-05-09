import torch
from torch import nn

from src.utils.quaternion import rotate_shape
# TODO: Check well two reductions per op, point-wise(n) and axis-wise(m)
REDUCTIONS = {
    "mean": torch.mean,
    "sum": torch.sum,
}


class RotationalSymmetryDistance(nn.Module):
    def __init__(self, p=1, reduction="mean"):
        super().__init__()
        self.p = p
        self.reduction = reduction

    def forward(self,
                points,
                axis_pred, axis_true,
                center_pred, center_true,
                angle_pred, angle_true
                ):
        m = axis_true.shape[0]
        n = points.shape[0]

        distances = torch.zeros(m, device=axis_true.device)

        for idx in range(m):
            rotated_pred = rotate_shape(axis_pred[idx], center_pred[idx], angle_pred[idx].reshape(1), points)
            rotated_true = rotate_shape(axis_true[idx], center_true[idx], angle_true[idx].reshape(1), points)
            distances[idx] = torch.norm(rotated_pred - rotated_true, p=self.p, dim=1).mean()

        return REDUCTIONS[self.reduction](distances)
