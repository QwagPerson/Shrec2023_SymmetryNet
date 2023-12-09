import math
import lightning
import torch
from lightning.pytorch.callbacks import EarlyStopping
from torch import nn, Tensor
from torch.utils.data import Subset, DataLoader

from src.metrics.mAP import get_mean_average_precision
from src.metrics.phc import get_matches_amount, get_phc
from src.model.center_n_normals.decoder.center_prediction_head import CenterPredictionHead
from src.model.center_n_normals.decoder.normal_prediction_head import NormalPredictionHead
from src.model.center_n_normals.encoder.pointnet_encoder import PointNetEncoder
from src.model.center_n_normals.loss.plane_nearness import calculate_loss
from src.model.center_n_normals.loss.symmetry_distance_error import calculate_loss_sde


class CenterNNormalsNet(nn.Module):
    def __init__(
            self,
            amount_of_normals_predicted: int = 3,
    ):
        super().__init__()

        self.h = amount_of_normals_predicted

        self.encoder = PointNetEncoder()
        self.normal_prediction_heads = nn.ModuleList(
            [NormalPredictionHead() for _ in range(self.h)]
        )

        self.center_prediction_head = CenterPredictionHead()

    def forward(self, x):
        batch_size = x.shape[0]
        normal_list = []

        x = self.encoder(x)

        for head in self.normal_prediction_heads:
            normal_list.append(head(x))

        center = self.center_prediction_head(x).unsqueeze(dim=1).repeat(1, self.h, 1)
        normals = torch.vstack(normal_list).view(batch_size, self.h, 4)

        predictions = torch.concat((center, normals), dim=2)

        predictions[:, :, -1] = torch.sigmoid(predictions[:, :, -1])
        predictions[:, :, 0:3] = torch.nn.functional.normalize(predictions[:, :, 0:3].clone(), dim=2)

        return predictions


class LightingCenterNNormalsNet(lightning.LightningModule):
    def __init__(self,
                 amount_of_normals_predicted: int = 3,
                 loss_fn=None,
                 ):
        super().__init__()
        self.net = CenterNNormalsNet(
            amount_of_normals_predicted=amount_of_normals_predicted,
        )
        if loss_fn is not None:
            self.loss_fn = loss_fn
        else:
            self.loss_fn = calculate_loss
        self.save_hyperparameters(ignore=["net", "loss_fn"])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        idxs, points, sym_planes, transforms = batch
        points = torch.transpose(points, 1, 2).float()
        y_pred = self.net.forward(points)
        loss = self.loss_fn(batch, y_pred)

        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=len(sym_planes))
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        idxs, points, sym_planes, transforms = batch
        points = torch.transpose(points, 1, 2).float()

        y_pred = self.net.forward(points)
        loss = self.loss_fn(batch, y_pred)

        self.log("val_loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=len(sym_planes))

        return loss

    def test_step(self, batch, batch_idx):
        idxs, points, sym_planes, transforms = batch
        points = torch.transpose(points, 1, 2).float()

        y_pred = self.net.forward(points)
        loss = self.loss_fn(batch, y_pred)

        self.log("test_loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=len(sym_planes))

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        idxs, points, y_true, transforms = batch
        points = torch.transpose(points, 1, 2).float()

        y_pred = self.net.forward(points)

        return batch, y_pred

    def on_after_backward(self):
        for name, param in self.net.named_parameters():
            if param.grad.isnan().any():
                print(f"{name} got nan!")


if __name__ == "__main__":
    from src.dataset.shrec2023 import (SymmetryDataset,
                                       SymmetryDataModule,
                                       default_symmetry_dataset_collate_fn,
                                       default_symmetry_dataset_collate_fn_list_sym)
    from src.dataset.preprocessing import *

    DATA_PATH = "/data/shrec_2023/benchmark-train"
    BATCH_SIZE = 3
    SAMPLE_SIZE = 1024
    COLLATE_FN = default_symmetry_dataset_collate_fn_list_sym
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
        collate_function=COLLATE_FN,
        validation_percentage=0.9999,
        shuffle=True,
        n_workers=1,
    )

    test_net = LightingCenterNNormalsNet(20, loss_fn=calculate_loss)  #
    mpath = ""
    #test_net = CenterNNormalsNet.load_from_checkpoint(mpath)

    datamodule.setup("predict")

    torch.autograd.set_detect_anomaly(True)
    trainer = lightning.Trainer(
        fast_dev_run=False,
        limit_val_batches=0.0,
        enable_progress_bar=True,
        max_epochs=500,
        callbacks=[
            EarlyStopping("train_loss", patience=10, verbose=True)
        ]
    )
    predict_dataset = Subset(datamodule.predict_dataset, [i for i in range(BATCH_SIZE)])
    predict_dataloader = DataLoader(predict_dataset, batch_size=BATCH_SIZE,
                                    collate_fn=COLLATE_FN)

    trainer.fit(test_net, predict_dataloader)
    predictions = trainer.predict(test_net, predict_dataloader)

    print(get_phc(predictions))

    print(get_mean_average_precision(predictions))



