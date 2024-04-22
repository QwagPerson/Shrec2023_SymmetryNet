#!/usr/bin/env python3
from typing import Callable

import random

import lightning
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch import nn
from torch.utils.data import Subset, DataLoader

from src.metrics.mAP import get_mean_average_precision, get_match_sequence
from src.metrics.phc import get_phc
from src.model.decoders.center_prediction_head import CenterPredictionHead
from src.model.decoders.normal_prediction_head import NormalPredictionHead
from src.model.encoders.pointnet_encoder import PointNetEncoder
from src.model.losses.discrete_prediction_loss import calculate_loss
from src.model.losses.utils import calculate_cost_matrix_normals, calculate_cost_matrix_sde
from src.model.postprocessing.utils import reverse_transformation


class CenterNNormalsNet(nn.Module):
    def __init__(
            self,
            amount_of_normals_predicted: int = 3,
            use_bn=False
    ):
        super().__init__()
        self.use_bn = use_bn
        self.h = amount_of_normals_predicted

        self.encoder = PointNetEncoder(use_bn=self.use_bn)
        self.normal_prediction_heads = nn.ModuleList(
            [NormalPredictionHead(use_bn=self.use_bn) for _ in range(self.h)]
        )

        self.center_prediction_head = CenterPredictionHead(use_bn=self.use_bn)

    def forward(self, x):
        batch_size = x.shape[0]
        normal_list = []

        x = self.encoder(x)

        for head in self.normal_prediction_heads:
            normal_list.append(head(x))

        center = self.center_prediction_head(x).unsqueeze(dim=1).repeat(1, self.h, 1)
        normals = torch.vstack(normal_list).view(batch_size, self.h, 4)  # Normal (3) + Confidence(1)

        predictions = torch.concat((normals, center), dim=2)
        reorder = torch.tensor([0, 1, 2, 4, 5, 6, 3], device=predictions.device).long()

        predictions = predictions[:, :, reorder]

        predictions[:, :, -1] = torch.sigmoid(predictions[:, :, -1])
        predictions[:, :, 0:3] = torch.nn.functional.normalize(predictions[:, :, 0:3].clone(), dim=2)

        return predictions


class LightingCenterNNormalsNet(lightning.LightningModule):
    def __init__(self,
                 amount_of_normals_predicted: int = 3,
                 confidence_loss_constant: float = 1.0,
                 sde_loss_constant: float = 1.0,
                 distance_loss_constant: float = 1.0,
                 angle_loss_constant: float = 1.0,
                 cost_matrix_method: Callable = calculate_cost_matrix_normals,
                 print_losses: bool = False,
                 use_bn: bool = False,
                 ):
        super().__init__()
        self.use_bn = use_bn
        self.print_losses = print_losses
        self.cost_matrix_method = cost_matrix_method
        self.losses_weights = torch.tensor([
            confidence_loss_constant,
            sde_loss_constant,
            distance_loss_constant,
            angle_loss_constant,
        ])

        self.net = CenterNNormalsNet(
            amount_of_normals_predicted=amount_of_normals_predicted,
            use_bn=self.use_bn
        )
        self.save_hyperparameters(ignore=["net"])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        idxs, points, sym_planes, transforms = batch
        points = torch.transpose(points, 1, 2).float()

        y_pred = self.net.forward(points)
        loss = calculate_loss(
            batch, y_pred,
            self.cost_matrix_method, self.losses_weights,
            self.print_losses
        )

        prediction = [(batch, y_pred)]
        mean_avg_precision = get_mean_average_precision(prediction)
        phc = get_phc(prediction)

        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=len(sym_planes))
        self.log("train_MAP", mean_avg_precision, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=len(sym_planes))
        self.log("train_PHC", phc, on_step=False, on_epoch=True,
                 prog_bar=False, sync_dist=True, batch_size=len(sym_planes))
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        idxs, points, sym_planes, transforms = batch
        points = torch.transpose(points, 1, 2).float()

        y_pred = self.net.forward(points)
        loss = calculate_loss(
            batch, y_pred,
            self.cost_matrix_method, self.losses_weights,
            self.print_losses
        )

        prediction = [(batch, y_pred)]
        mean_avg_precision = get_mean_average_precision(prediction)
        phc = get_phc(prediction)

        self.log("val_loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=len(sym_planes))
        self.log("val_MAP", mean_avg_precision, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=len(sym_planes))
        self.log("val_PHC", phc, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=len(sym_planes))

        return loss

    def test_step(self, batch, batch_idx):
        idxs, points, sym_planes, transforms = batch
        points = torch.transpose(points, 1, 2).float()

        y_pred = self.net.forward(points)
        loss = calculate_loss(
            batch, y_pred,
            self.cost_matrix_method, self.losses_weights,
            self.print_losses
        )

        prediction = [(batch, y_pred)]
        mean_avg_precision = get_mean_average_precision(prediction)
        phc = get_phc(prediction)

        unscaled_batch, unscaled_y_pred = reverse_transformation(batch, y_pred)

        unscaled_prediction = [(unscaled_batch, unscaled_y_pred)]
        unscaled_mean_avg_precision = get_mean_average_precision(unscaled_prediction)
        unscaled_phc = get_phc(unscaled_prediction)

        self.log("test_loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=len(sym_planes))
        self.log("test_MAP", mean_avg_precision, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=len(sym_planes))
        self.log("test_PHC", phc, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=len(sym_planes))
        self.log("unscaled_test_MAP", unscaled_mean_avg_precision, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=len(sym_planes))
        self.log("unscaled_test_PHC", unscaled_phc, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=len(sym_planes))
        return batch, y_pred

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
                                       default_symmetry_dataset_collate_fn_list_sym)
    from src.dataset.preprocessing import *

    DATA_PATH = "/data/shrec_2023/benchmark-train"
    TEST_PATH = "/data/shrec_2023/benchmark-train"
    #DATA_PATH = "/tmp/ramdrive/benchmark-train-14400"
    #TEST_PATH = "/tmp/ramdrive/benchmark-train-14400"
    BATCH_SIZE = 1
    PREDICT_SAMPLES = 64
    SAMPLE_SIZE = 14_440
    COLLATE_FN = default_symmetry_dataset_collate_fn_list_sym
    NUM_WORKERS = 15

    scaler = UnitSphereNormalization()
    #sampler = RandomSampler(sample_size=SAMPLE_SIZE, keep_copy=True)
    #compose_transform = ComposeTransform([sampler, scaler])
    compose_transform = scaler

    dataset = SymmetryDataset(DATA_PATH, compose_transform)
    datamodule = SymmetryDataModule(
        train_data_path=DATA_PATH,
        test_data_path=TEST_PATH,
        predict_data_path=TEST_PATH,
        does_predict_has_ground_truths=True,
        batch_size=BATCH_SIZE,
        transform=compose_transform,
        collate_function=COLLATE_FN,
        validation_percentage=0.2,
        shuffle=True,
        n_workers=NUM_WORKERS,
    )
    datamodule.setup("predict")
    datamodule.setup("fit")

    #test_model = '/mnt/btrfs-data/venvs/ml-tutorials/repos/pointnet/Shrec2023_SymmetryNet-orig/epoch_epoch=1_val_MAP=0.56_train_MAP=0.46.ckpt'
    test_model = "models/bs4/epoch=37_train_loss_epoch=1.52210_val_MAP=0.39_train_MAP=0.38.ckpt"
    test_net = LightingCenterNNormalsNet.load_from_checkpoint(test_model)
    trainer = lightning.Trainer(enable_progress_bar=True, logger=False)

    print(f'Training dataset has: {len(datamodule.train_dataset) = } batches')
    train_dataloader = datamodule.train_dataloader()
    print(f'Train dataloader has: {len(train_dataloader) = } batches')
    val_dataloader = datamodule.val_dataloader()
    print(f'Valid dataloader has: {len(val_dataloader) = } batches')

    print(f'Predict dataset has: {len(datamodule.predict_dataset) = } batches')
    predict_dataset = Subset(datamodule.predict_dataset, [i for i in range(1, PREDICT_SAMPLES)])
    print(f'Predict dataset has: {len(predict_dataset) = } batches')
    predict_dataloader = DataLoader(predict_dataset, batch_size=BATCH_SIZE,
                                    collate_fn=COLLATE_FN, num_workers=NUM_WORKERS)
    print(f'Predict dataloader has: {len(predict_dataloader) = } batches')

    predictions = trainer.predict(test_net, predict_dataloader)

    print(f'Predictions: {len(predictions)}')
    pr_idx = random.randint(0, len(predictions) - 1)
    print(f'Taking batch no. {pr_idx}')
    batch, y_pred = predictions[pr_idx]
    _, points, y_true, _ = batch

    torch.set_printoptions(linewidth=200)
    torch.set_printoptions(precision=3)
    torch.set_printoptions(sci_mode=False)

    print(f"Batch size: {len(y_true)}")
    idx = random.randint(0, len(y_true) - 1)
    print(f"Comparing element: {idx} in batch...")
    gt = y_true[idx]
    pr = y_pred[idx][y_pred[idx][:, -1].sort(descending=True).indices]
    match_sequence = get_match_sequence(pr, gt, points[idx], eps=0.01, theta=0.0174533)

    print(f'Ground truth:\n{gt}')
    print(f'Prediction  :\n{pr}')
    print(f'Match Sequence: \n{match_sequence}')

    print("Normal metrics")
    print("PHC", get_phc(predictions).item())
    print("MAP", get_mean_average_precision(predictions).item())

    print("Normals matching")
    print("PHC", get_phc(predictions, eps=1).item())
    print("MAP", get_mean_average_precision(predictions, eps=1).item())

    print("Center matching")
    print("PHC", get_phc(predictions, theta=10).item())
    print("MAP", get_mean_average_precision(predictions, theta=10).item())
