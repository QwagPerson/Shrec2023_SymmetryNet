import math

import torch
from torch import nn

from src.model.backbones.pointnet_encoder import PointNet
from src.model.swp import Swp1d


class SymmetryNet(nn.Module):
    def __init__(self, batch_size, num_points: int, n_prediction_per_point: int = 15):
        super().__init__()

        self.encoder = PointNet()
        self.batch_size = batch_size
        self.num_points = num_points
        self.swp = Swp1d(batch_size, num_points, 1)
        self.k = n_prediction_per_point

        self.decoder = torch.nn.Sequential(
            nn.Conv1d(640, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 7 * self.k, 1),
            nn.BatchNorm1d(7 * self.k),
        )

    def forward(self, x):
        local_features, _, _ = self.encoder(x)
        #print("local_feat", local_features.shape)

        global_features = self.swp(local_features)
        #print("global_feat", global_features.shape)

        # combining global and local
        x = torch.concat([global_features.repeat(1, 1, local_features.shape[2]), local_features], dim=1)
        #print("features_concat", x.shape)

        x = self.decoder(x)
        #print("out_shape", x.shape)
        x = x.view(self.batch_size, self.num_points, self.k, 7)

        m = nn.Sigmoid()

        x[:, :, :, -1] = m(x[:, :, :, -1])

        return x


if __name__ == "__main__":
    from src.dataset.shrec2023 import (SymmetryDataset,
                                       SymmetryDataModule,
                                       default_symmetry_dataset_collate_fn)
    from src.dataset.preprocessing import *
    from src.model.loss import calculate_loss

    DATA_PATH = "/data/shrec_2023/benchmark-train"
    BATCH_SIZE = 2
    SAMPLE_SIZE = 10

    scaler = UnitSphereNormalization()
    sampler = RandomSampler(sample_size=SAMPLE_SIZE, keep_copy=True)
    compose_transform = ComposeTransform([scaler, sampler])

    dataset = SymmetryDataset(DATA_PATH, compose_transform)

    datamodule = SymmetryDataModule(
        train_data_path=DATA_PATH,
        test_data_path=DATA_PATH,
        predict_data_path=DATA_PATH,
        does_predict_has_ground_truths=True,
        batch_size=BATCH_SIZE,
        transform=compose_transform,
        collate_function=default_symmetry_dataset_collate_fn,
        validation_percentage=0.1,
        shuffle=True,
        n_workers=1,
    )
    datamodule.setup("fit")

    train_dataloader = datamodule.train_dataloader()
    batch = next(iter(train_dataloader))
    idxs, points, sym_planes, transforms = batch
    bs, np, npp = BATCH_SIZE, SAMPLE_SIZE, 23

    net = SymmetryNet(batch_size=bs, num_points=np, n_prediction_per_point=npp)
    optimizer = torch.optim.Adam(net.parameters())

    #print(points.shape)
    points = torch.transpose(points, 1, 2).float()
    #print(points.shape)
    out = net.forward(points)
    #print(out.shape)
    loss = calculate_loss(batch, out)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

