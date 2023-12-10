import torch.nn
from torch import nn


class NormalPredictionHead(nn.Module):
    def __init__(self, use_bn=False):
        super().__init__()
        self.use_bn = use_bn
        if use_bn:
            self.decoder_head = torch.nn.Sequential(
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(),
                nn.Linear(256, 4),
            )
        else:
            self.decoder_head = torch.nn.Sequential(
                nn.Linear(1024, 512),
                nn.LeakyReLU(),
                nn.Linear(512, 256),
                nn.LeakyReLU(),
                nn.Linear(256, 4),
            )


    def forward(self, x):
        """
        :param x: B x C
        :return: y: B x 4
        """
        return self.decoder_head(x)
