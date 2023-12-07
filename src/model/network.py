import math
import lightning
import torch
from torch import nn

from src.model.encoder.pointnet_encoder import PointNetEncoder
from src.model.loss.loss import calculate_loss
from src.model.postprocessing import postprocess_predictions


class SymmetryNet(nn.Module):
    def __init__(self,
                 batch_size,
                 num_points: int = 1000,
                 n_prediction_per_point: int = 15,

                 ):
        super().__init__()

        self.batch_size = batch_size
        self.num_points = num_points
        self.k = n_prediction_per_point

        self.encoder = PointNetEncoder(batch_size, num_points)
        self.decoder = torch.nn.Sequential(
            nn.Conv1d(512, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 7 * self.k, 1),
            nn.BatchNorm1d(7 * self.k),
        )

    def forward(self, x):
        # print("original", x.shape)
        x = self.encoder(x)
        # print("encoded", x.shape)
        x = self.decoder(x)
        # print("decoded", x.shape)
        x = x.view(self.batch_size, self.num_points, self.k, 7)

        m = nn.Sigmoid()
        x[:, :, :, -1] = m(x[:, :, :, -1])

        return x


class LightingSymmetryNet(lightning.LightningModule):
    def __init__(self,
                 batch_size: int = 2,
                 num_points: int = 1000,
                 n_prediction_per_point: int = 20,
                 dbscan_eps: float = 0.2,
                 dbscan_min_samples: int = 500,
                 n_jobs: int = 4,
                 ):
        super().__init__()
        self.net = SymmetryNet(batch_size, num_points, n_prediction_per_point)
        self.loss_fn = calculate_loss
        self.batch_size = batch_size
        self.num_points = num_points
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.n_jobs = n_jobs
        self.save_hyperparameters(ignore=["net", "loss_fn"])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        idxs, points, sym_planes, transforms = batch
        points = torch.transpose(points, 1, 2).float()
        y_pred = self.net.forward(points)
        loss = self.loss_fn(batch, y_pred)

        self.log("train_loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=self.batch_size)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        idxs, points, sym_planes, transforms = batch
        points = torch.transpose(points, 1, 2).float()

        y_pred = self.net.forward(points)
        loss = self.loss_fn(batch, y_pred)

        self.log("val_loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=self.batch_size)

        return loss

    def test_step(self, batch, batch_idx):
        idxs, points, sym_planes, transforms = batch
        points = torch.transpose(points, 1, 2).float()

        y_pred = self.net.forward(points)
        loss = self.loss_fn(batch, y_pred)

        self.log("test_loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=self.batch_size)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        idxs, points, y_true, transforms = batch
        points = torch.transpose(points, 1, 2).float()

        y_pred = self.net.forward(points)
        prediction = postprocess_predictions(
            y_pred,
            eps=self.dbscan_eps,
            min_samples=self.dbscan_min_samples,
            n_jobs=self.n_jobs
        )
        return prediction, y_true


if __name__ == "__main__":
    from src.dataset.shrec2023 import (SymmetryDataset,
                                       SymmetryDataModule,
                                       default_symmetry_dataset_collate_fn)
    from src.dataset.preprocessing import *

    DATA_PATH = "/data/shrec_2023/benchmark-train"
    BATCH_SIZE = 1
    SAMPLE_SIZE = 1000

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
        validation_percentage=0.9999,
        shuffle=True,
        n_workers=1,
    )

    lnet = LightingSymmetryNet(BATCH_SIZE, SAMPLE_SIZE, 10)
    trainer = lightning.Trainer(fast_dev_run=False, limit_val_batches=0.0, enable_progress_bar=True)
