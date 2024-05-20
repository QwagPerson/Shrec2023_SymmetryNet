from torch import nn


class ConfidenceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, conf_pred, conf_true):
        return nn.functional.binary_cross_entropy(conf_pred, conf_true)
