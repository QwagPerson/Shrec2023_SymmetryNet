import torch
from torch import nn


class ConfidenceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.weighted = True

    def forward(self, conf_pred, conf_true):
        if self.weighted:
            h = conf_pred.shape[0]
            r = conf_true.sum().item()
            p1 = (h - r) / h
            p2 = r / h
            weights = (p1 / r) * conf_true + (1 - conf_true) * (p2 / h)
            return nn.functional.binary_cross_entropy(conf_pred, conf_true, weights)
        else:
            return nn.functional.binary_cross_entropy(conf_pred, conf_true)
