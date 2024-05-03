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
from src.model.decoders.prediction_head import PredictionHead
from src.model.encoders.pointnet_encoder import PointNetEncoder
from src.model.losses.discrete_prediction_loss import calculate_loss
from src.model.losses.utils import calculate_cost_matrix_normals, calculate_cost_matrix_sde
from src.model.postprocessing.utils import reverse_transformation


class CenterNNormalsNet(nn.Module):
    def __init__(
            self,
            amount_of_plane_normals_predicted=10,
            amount_of_axis_discrete_normals_predicted=10,
            amount_of_axis_continue_normals_predicted=10,
            use_bn=False,
            normalize_normals=False,
    ):
        super().__init__()
        self.use_bn = use_bn
        self.normalize_normals = normalize_normals
        self.amount_plane_normals = amount_of_plane_normals_predicted
        self.amount_axis_discrete_normals = amount_of_axis_discrete_normals_predicted
        self.amount_axis_continue_normals = amount_of_axis_continue_normals_predicted

        self.encoder = PointNetEncoder(use_bn=self.use_bn)

        # nx ny nz & confidence
        self.plane_normals_heads = nn.ModuleList(
            [PredictionHead(output_size=4, use_bn=self.use_bn) for _ in range(self.amount_plane_normals)]
        )

        # nx ny nz theta & confidence
        self.axis_discrete_normals_heads = nn.ModuleList(
            [PredictionHead(output_size=5, use_bn=self.use_bn) for _ in range(self.amount_axis_discrete_normals)]
        )

        # nx ny nz & confidence
        self.axis_continue_normals_heads = nn.ModuleList(
            [PredictionHead(output_size=4, use_bn=self.use_bn) for _ in range(self.amount_axis_continue_normals)]
        )

        self.center_prediction_head = CenterPredictionHead(use_bn=self.use_bn)

    def forward(self, x):
        batch_size = x.shape[0]
        plane_normals_list = []
        axis_discrete_normals_list = []
        axis_continue_normals_list = []

        x = self.encoder(x)
        center = self.center_prediction_head(x).unsqueeze(dim=1)

        for head in self.plane_normals_heads:
            plane_normals_list.append(head(x))

        for head in self.axis_discrete_normals_heads:
            axis_discrete_normals_list.append(head(x))

        for head in self.axis_continue_normals_heads:
            axis_continue_normals_list.append(head(x))

        # Plane prediction
        # Normal (3) + Confidence(1)
        if self.amount_plane_normals > 0:
            plane_normals = (torch.vstack(plane_normals_list).view(
                batch_size, self.amount_plane_normals, 4
            ))
            plane_predictions = torch.concat(
                (plane_normals, center.repeat(1, self.amount_plane_normals, 1)), dim=2
            )

            reorder_planes = torch.tensor([0, 1, 2, 4, 5, 6, 3], device=plane_predictions.device).long()
            plane_predictions = plane_predictions[:, :, reorder_planes]
            plane_predictions[:, :, -1] = torch.sigmoid(plane_predictions[:, :, -1])
        else:
            plane_predictions = None

        # Axis discrete prediction
        # Normal (3) + Theta (1) + Confidence(1)
        if self.amount_axis_discrete_normals > 0:
            axis_discrete_normals = (torch.vstack(axis_discrete_normals_list).view(
                batch_size, self.amount_axis_discrete_normals, 5
            ))
            axis_discrete_predictions = torch.concat(
                (axis_discrete_normals, center.repeat(1, self.amount_axis_discrete_normals, 1)), dim=2
            )
            reorder_axis_discrete = torch.tensor([0, 1, 2, 4, 5, 6, 7, 3], device=plane_predictions.device).long()
            axis_discrete_predictions = axis_discrete_predictions[:, :, reorder_axis_discrete]
            axis_discrete_predictions[:, :, -1] = torch.sigmoid(axis_discrete_predictions[:, :, -1])
        else:
            axis_discrete_predictions = None

        # Axis continue prediction
        # Normal (3) + Confidence(1)
        if self.amount_axis_continue_normals > 0:
            axis_continue_normals = (torch.vstack(axis_continue_normals_list).view(
                batch_size, self.amount_axis_continue_normals, 4
            ))
            axis_continue_predictions = torch.concat(
                (axis_continue_normals, center.repeat(1, self.amount_axis_continue_normals, 1)), dim=2
            )

            reorder_planes = torch.tensor([0, 1, 2, 4, 5, 6, 3], device=axis_continue_predictions.device).long()
            axis_continue_predictions = axis_continue_predictions[:, :, reorder_planes]
            axis_continue_predictions[:, :, -1] = torch.sigmoid(axis_continue_predictions[:, :, -1])
        else:
            axis_continue_predictions = None

        if self.normalize_normals:
            if plane_predictions is not None:
                plane_predictions[:, :, 0:3] = torch.nn.functional.normalize(
                    plane_predictions[:, :, 0:3].clone(), dim=2
                )
            if axis_discrete_predictions is not None:
                axis_discrete_predictions[:, :, 0:3] = torch.nn.functional.normalize(
                    axis_discrete_predictions[:, :, 0:3].clone(), dim=2
                )
            if axis_continue_predictions is not None:
                axis_continue_predictions[:, :, 0:3] = torch.nn.functional.normalize(
                    axis_continue_predictions[:, :, 0:3].clone(), dim=2
                )

        return plane_predictions, axis_discrete_predictions, axis_continue_predictions


class LightingCenterNNormalsNet(lightning.LightningModule):
    def __init__(self,
                 amount_of_plane_normals_predicted: int = 32,
                 amount_of_axis_discrete_normals_predicted: int = 16,
                 amount_of_axis_continue_normals_predicted: int = 16,
                 confidence_loss_constant: float = 1.0,
                 sde_loss_constant: float = 1.0,
                 distance_loss_constant: float = 1.0,
                 angle_loss_constant: float = 1.0,
                 cost_matrix_method: Callable = calculate_cost_matrix_normals,
                 print_losses: bool = False,
                 use_bn: bool = False,
                 normalize_normals: bool = True
                 ):
        super().__init__()
        self.use_bn = use_bn
        self.normalize_normals = normalize_normals
        self.print_losses = print_losses
        self.cost_matrix_method = cost_matrix_method
        self.losses_weights = torch.tensor([
            confidence_loss_constant,
            sde_loss_constant,
            distance_loss_constant,
            angle_loss_constant,
        ])

        self.net = CenterNNormalsNet(
            amount_of_plane_normals_predicted,
            amount_of_axis_discrete_normals_predicted,
            amount_of_axis_continue_normals_predicted,
            use_bn=self.use_bn,
            normalize_normals=self.normalize_normals
        )
        self.save_hyperparameters(ignore=["net"])

    def configure_optimizers(self):
        # Does this matter much?? self.parameters() vs self.net.parameters()
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        idxs, points, planar_syms, axis_continue_syms, axis_discrete_syms, transforms = batch
        points = torch.transpose(points, 1, 2).float()

        plane_predictions, axis_discrete_predictions, axis_continue_predictions = self.net.forward(points)
        loss = calculate_loss(
            batch, plane_predictions,
            self.cost_matrix_method, self.losses_weights,
            self.print_losses
        )

        prediction = [(batch, plane_predictions)]
        mean_avg_precision = get_mean_average_precision(prediction)
        phc = get_phc(prediction)

        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=len(planar_syms))
        self.log("train_MAP", mean_avg_precision, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=len(planar_syms))
        self.log("train_PHC", phc, on_step=False, on_epoch=True,
                 prog_bar=False, sync_dist=True, batch_size=len(planar_syms))
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        idxs, points, planar_syms, axis_continue_syms, axis_discrete_syms, transforms = batch
        points = torch.transpose(points, 1, 2).float()

        plane_predictions, axis_discrete_predictions, axis_continue_predictions = self.net.forward(points)
        loss = calculate_loss(
            batch, plane_predictions,
            self.cost_matrix_method, self.losses_weights,
            self.print_losses
        )

        prediction = [(batch, plane_predictions)]
        mean_avg_precision = get_mean_average_precision(prediction)
        phc = get_phc(prediction)

        self.log("val_loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=len(planar_syms))
        self.log("val_MAP", mean_avg_precision, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=len(planar_syms))
        self.log("val_PHC", phc, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=len(planar_syms))

        return loss

    def test_step(self, batch, batch_idx):
        idxs, points, planar_syms, axis_continue_syms, axis_discrete_syms, transforms = batch
        points = torch.transpose(points, 1, 2).float()

        plane_predictions, axis_discrete_predictions, axis_continue_predictions = self.net.forward(points)
        loss = calculate_loss(
            batch, plane_predictions,
            self.cost_matrix_method, self.losses_weights,
            self.print_losses
        )

        prediction = [(batch, plane_predictions)]
        mean_avg_precision = get_mean_average_precision(prediction)
        phc = get_phc(prediction)

        unscaled_batch, unscaled_plane_predictions = reverse_transformation(batch, plane_predictions)

        unscaled_prediction = [(unscaled_batch, unscaled_plane_predictions)]
        unscaled_mean_avg_precision = get_mean_average_precision(unscaled_prediction)
        unscaled_phc = get_phc(unscaled_prediction)

        self.log("test_loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=len(planar_syms))
        self.log("test_MAP", mean_avg_precision, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=len(planar_syms))
        self.log("test_PHC", phc, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=len(planar_syms))
        self.log("unscaled_test_MAP", unscaled_mean_avg_precision, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=len(planar_syms))
        self.log("unscaled_test_PHC", unscaled_phc, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=len(planar_syms))
        return batch, plane_predictions

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        idxs, points, planar_syms, axis_continue_syms, axis_discrete_syms, transforms = batch
        points = torch.transpose(points, 1, 2).float()

        planar_syms = self.net.forward(points)

        return batch, planar_syms

    def on_after_backward(self):
        for name, param in self.net.named_parameters():
            if param.grad is not None:
                if param.grad.isnan().any():
                    print(f"{name} got nan!")


if __name__ == "__main__":
    from src.dataset.shrec2023 import (SymmetryDataset,
                                       SymmetryDataModule,
                                       default_symmetry_dataset_collate_fn_list_sym)
    from src.dataset.preprocessing import *

    DATA_PATH = "/data/sym-10k-xz-split-class-noparallel/"
    BATCH_SIZE = 3
    PREDICT_SAMPLES = 1
    SAMPLE_SIZE = 14_440
    COLLATE_FN = default_symmetry_dataset_collate_fn_list_sym
    NUM_WORKERS = 15

    scaler = UnitSphereNormalization()
    sampler = RandomSampler(sample_size=SAMPLE_SIZE, keep_copy=True)
    compose_transform = ComposeTransform([sampler, scaler])

    datamodule = SymmetryDataModule(
        dataset_path=DATA_PATH,
        predict_data_path=DATA_PATH,
        does_predict_has_ground_truths=True,
        batch_size=BATCH_SIZE,
        transform=compose_transform,
        collate_function=default_symmetry_dataset_collate_fn_list_sym,
        shuffle=True,
        n_workers=1,
    )
    datamodule.setup("predict")
    datamodule.setup("fit")

    test_net = LightingCenterNNormalsNet(amount_of_plane_normals_predicted=10,
                                         amount_of_axis_discrete_normals_predicted=0,
                                         amount_of_axis_continue_normals_predicted=0, )
    trainer = lightning.Trainer(enable_progress_bar=True, logger=False)

    test_batch = next(iter(datamodule.train_dataloader()))

    trainer.fit(test_net, datamodule)

    """
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
    """
