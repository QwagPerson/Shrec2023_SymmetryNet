import torch
from torch import nn

REDUCTIONS = {
    "mean": torch.mean,
    "sum": torch.sum,
}


class DistanceLoss(nn.Module):
    def __init__(self, p=1, reduction="mean"):
        super().__init__()
        self.p = p
        self.reduction = reduction

    def forward(self, center_pred, center_true):
        # Should here dim=1?? as Nx3, Nx3 -> N instead of Nx3, Nx3 -> 3
        # IM CHANGIN THIS TODO
        distances = torch.norm(center_true - center_pred, p=1, dim=0)
        return REDUCTIONS[self.reduction](distances)


if __name__ == "__main__":
    a, b = torch.rand((3, 3)), torch.rand((3, 3))

    old = torch.norm(a - b, p=1, dim=1).mean()
    new = DistanceLoss(p=1).forward(a, b)
    print(old, new)
