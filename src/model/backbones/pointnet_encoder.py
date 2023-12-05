import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import numpy as np
import os
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from tqdm import tqdm
from torch.autograd import Variable
import math


class TNet(nn.Module):
    """
    Transform Net: Network that learns a transformation matrix

    Args:
        in_dim (int): Input dimension
        k (int): Size of transformation matrix
    """

    def __init__(self, in_dim: int = 3, k: int = 3):
        super(TNet, self).__init__()
        self.in_dim = in_dim
        self.k = k

        self.shared_mlps = torch.nn.Sequential(
            nn.Conv1d(in_dim, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )

        self.linear = torch.nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, k * k),
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass
        :param x: Tensor of shape (batch_size, in_dim, num_points)
        :return: y: Tensor of shape (batch_size, out_dim)
        """
        batch_size = x.size()[0]
        x = torch.max(self.shared_mlps(x), dim=2).values
        x = self.linear(x)
        identity = Variable(
            torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))
        ).view(1, self.k * self.k).repeat(batch_size, 1)
        if x.is_cuda:
            identity = identity.cuda()
        x = x + identity
        return x


class PointNet(nn.Module):
    """
    PointNet: Implementation of PointNet that calculates the global features.
    """

    def __init__(self):
        super(PointNet, self).__init__()

        self.input_transform = TNet(in_dim=3, k=3)
        self.shared_mlps = torch.nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.shared_mlps_2 = torch.nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
        )

        self.feature_transform = TNet(in_dim=64, k=64)

    def forward(self, x0: torch.Tensor):
        """
        Forward pass
        :param x: Tensor of shape (batch_size, 3, num_points)
        :return:
            local_features: Tensor of shape (batch_size, 1024, num_points)
            global_features: Tensor of shape (batch_size, 1024)

        """

        input_trans = self.input_transform(x0)
        input_trans = input_trans.reshape(-1, 3, 3)

        x0 = x0.transpose(2, 1)
        x0 = torch.bmm(x0, input_trans)
        x0 = x0.transpose(2, 1)

        x1 = self.shared_mlps(x0)

        feat_trans = self.feature_transform(x1)
        feat_trans = feat_trans.reshape(-1, 64, 64)

        x1 = x1.transpose(2, 1)
        x1 = torch.bmm(x1, feat_trans)
        x1 = x1.transpose(2, 1)

        x2 = self.shared_mlps_2(x1)

        local_features = torch.cat([x1, x2], dim=1)

        return local_features, input_trans, feat_trans
