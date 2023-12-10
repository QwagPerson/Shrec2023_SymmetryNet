import torch.nn
from torch import nn


class DensePredictionDecoder(nn.Module):
    """
    Transforms the output of a PointNetEncoder into
    B x 512 x S -> B x 7 x S
    """

    def __init__(self, n_pred_per_point: int):
        super().__init__()

        self.m = n_pred_per_point

        self.decoder_head = torch.nn.Sequential(
            nn.Conv1d(1024, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.ReLU(),
        )

        self.center_estimator = torch.nn.Sequential(
            nn.Conv1d(128, 3, 1),
        )

        self.normals_estimator = torch.nn.Sequential(
            nn.Conv1d(128, 3 * n_pred_per_point, 1),
        )

        self.confidence_estimator = torch.nn.Sequential(
            nn.Conv1d(128, n_pred_per_point, 1),
        )

    def forward(self, x):
        """

        :param x: B x C x N
        :return: center_pred      -> B x N x 3
                 normals_pred     -> B x N x M x 3
                 confidences_pred -> B x N x M
        """
        b, c, n = x.shape
        x = self.decoder_head(x)
        center_pred = self.center_estimator(x).view(b, n, 3)
        normals_pred = self.normals_estimator(x).view(b, n, self.m, 3)
        confidences_pred = self.confidence_estimator(x).view(b, n, self.m)
        return center_pred, normals_pred, confidences_pred
