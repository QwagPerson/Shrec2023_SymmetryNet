from torch import nn


class ConfidenceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.binary_cross_entropy = nn.BCELoss()

    def forward(self, conf_pred, conf_true):
        return self.binary_cross_entropy(input=conf_pred, target=conf_true)
