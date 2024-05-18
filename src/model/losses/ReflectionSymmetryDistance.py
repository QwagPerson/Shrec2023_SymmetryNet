import torch
from torch import nn
from torch import linalg as alg

REDUCTIONS = {
    "mean": torch.mean,
    "sum": torch.sum,
}


class ReflectionSymmetryDistance(nn.Module):
    def __init__(self, p=1, reduction="mean"):
        super().__init__()
        self.p = p
        self.reduction = reduction

    def forward(self,
                points,
                normal_pred, normal_true,
                center_pred, center_true
                ):
        m = normal_true.shape[0]
        n = points.shape[0]

        points = points.repeat(m, 1)

        normal_true = normal_true.repeat(n, 1)
        normal_pred = normal_pred.repeat(n, 1)

        center_true = center_true.repeat(n, 1)
        center_pred = center_pred.repeat(n, 1)

        offset_true = - alg.vecdot(normal_true, center_true)
        offset_pred = - alg.vecdot(normal_pred, center_pred)

        distances_true = alg.vecdot(points, normal_true) + offset_true
        distances_pred = alg.vecdot(points, normal_pred) + offset_pred

        distances_true = distances_true.unsqueeze(-1).repeat(1, 3)
        distances_pred = distances_pred.unsqueeze(-1).repeat(1, 3)

        reflected_true = points - 2 * distances_true * normal_true
        reflected_pred = points - 2 * distances_pred * normal_pred

        distance_between_reflected = torch.norm(reflected_pred - reflected_true, p=self.p, dim=1)

        return REDUCTIONS[self.reduction](distance_between_reflected)


