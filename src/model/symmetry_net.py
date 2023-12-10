import lightning
from torch import nn

from src.model.decoders.dense_prediction_decoder import DensePredictionDecoder
from src.model.encoders.dense_pointnet_encoder import DensePointNetEncoder
from src.model.losses.dense_prediction_loss import calculate_loss


class SymmetryNet(nn.Module):
    def __init__(self,
                 batch_size,
                 num_points: int = 1000,
                 n_heads: int = 3,
                 n_prediction_per_point: int = 3,

                 ):
        super().__init__()

        self.batch_size = batch_size
        self.num_points = num_points
        self.k = n_prediction_per_point
        self.h = n_heads

        self.encoder = DensePointNetEncoder(batch_size, num_points)
        self.decoders = nn.ModuleList(
            [DensePredictionDecoder(self.k) for _ in range(self.h)]
        )

    def forward(self, x):
        x = self.encoder(x)
        pred_list = []
        for decoder in self.decoders:
            pred_list.append(decoder(x))
        return pred_list


class LightingSymmetryNet(lightning.LightningModule):
    def __init__(self,
                 batch_size: int = 2,
                 num_points: int = 1000,
                 n_prediction_per_point: int = 20,
                 n_heads: int = 3,
                 dbscan_eps: float = 0.2,
                 dbscan_min_samples: int = 500,
                 n_jobs: int = 4,
                 loss_fn=calculate_loss,
                 ):
        super().__init__()
        self.net = SymmetryNet(
            batch_size=batch_size,
            num_points=num_points,
            n_prediction_per_point=n_prediction_per_point,
            n_heads=n_heads,
        )
        self.loss_fn = loss_fn
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

        return y_pred, y_true


if __name__ == "__main__":
    from src.dataset.shrec2023 import (SymmetryDataset,
                                       SymmetryDataModule,
                                       default_symmetry_dataset_collate_fn,
                                       default_symmetry_dataset_collate_fn_list_sym)
    from src.dataset.preprocessing import *

    DATA_PATH = "/data/shrec_2023/benchmark-train"
    BATCH_SIZE = 2
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
        collate_function=default_symmetry_dataset_collate_fn_list_sym,
        validation_percentage=0.99,
        shuffle=True,
        n_workers=0,
    )

    lnet = LightingSymmetryNet(BATCH_SIZE, SAMPLE_SIZE, 10)
    trainer = lightning.Trainer(fast_dev_run=False, limit_val_batches=0.0, enable_progress_bar=True)

    trainer.fit(lnet, datamodule)
