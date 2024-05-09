from typing import Callable

import lightning
import torch

from src.metrics.MAP import get_mean_average_precision
from src.metrics.PHC import get_phc
from src.model.CenterNNormalsNet import CenterNNormalsNet
from src.model.losses.DiscreteRotationalSymmetryLoss import DiscreteRotationalSymmetryLoss
from src.model.losses.NormalLoss import NormalLoss
from src.model.losses.ConfidencesLoss import ConfidenceLoss
from src.model.losses.DistanceLoss import DistanceLoss
from src.model.losses.ReflectionSymmetryDistance import ReflectionSymmetryDistance
from src.model.losses.ReflectionSymmetryLoss import ReflectionSymmetryLoss
from src.model.losses.RotationalSymmetryDistance import RotationalSymmetryDistance
from src.model.losses.RotationalSymmetryLoss import RotationalSymmetryLoss
from src.model.losses.utils import reverse_transformation
from src.model.matchers.SimpleMatcher import SimpleMatcher
from src.model.matchers.cost_matrix_methods import calculate_cost_matrix_normals


class LightingCenterNNormalsNet(lightning.LightningModule):
    def __init__(self,
                 amount_of_plane_normals_predicted: int = 32,
                 amount_of_axis_discrete_normals_predicted: int = 16,
                 amount_of_axis_continue_normals_predicted: int = 16,
                 conf_weight: float = 1.0,
                 sym_dist_weight: float = 1.0,
                 dist_weight: float = 1.0,
                 normal_weight: float = 1.0,
                 angle_weight: float = 1.0,
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
        self.matcher = SimpleMatcher(self.cost_matrix_method)

        self.plane_loss = ReflectionSymmetryLoss(
            confidence_weight=conf_weight, confidence_loss=ConfidenceLoss(),
            normal_weight=normal_weight, normal_loss=NormalLoss(),
            distance_weight=dist_weight, distance_loss=DistanceLoss(),
            reflection_symmetry_distance_weight=sym_dist_weight,
            reflection_symmetry_distance=ReflectionSymmetryDistance()
        )

        self.discrete_rotational_loss = DiscreteRotationalSymmetryLoss(
            confidence_weight=conf_weight, confidence_loss=ConfidenceLoss(),
            normal_weight=normal_weight, normal_loss=NormalLoss(),
            distance_weight=dist_weight, distance_loss=DistanceLoss(),
            angle_weight=angle_weight, angle_loss=DistanceLoss(),
            rotational_symmetry_distance_weight=sym_dist_weight,
            rotational_symmetry_distance=RotationalSymmetryDistance()
        )

        self.continue_rotational_loss = RotationalSymmetryLoss(
            confidence_weight=conf_weight, confidence_loss=ConfidenceLoss(),
            normal_weight=normal_weight, normal_loss=NormalLoss(),
            distance_weight=dist_weight, distance_loss=DistanceLoss(),
            angle_weight=angle_weight, angle_loss=DistanceLoss(),
            rotational_symmetry_distance_weight=sym_dist_weight,
            rotational_symmetry_distance=RotationalSymmetryDistance()
        )

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

    def _process_prediction(self, batch, sym_pred, sym_true, loss_fun, sym_tag, step_tag):
        c_hat, match_pred, match_true = self.matcher.get_optimal_assignment(batch.get_points(), sym_pred, sym_true)
        bundled_predictions = (batch, sym_pred, c_hat, match_pred, match_true)
        loss = loss_fun(bundled_predictions)

        eval_predictions = [(batch.get_points(), sym_pred, sym_true)]
        map = get_mean_average_precision(eval_predictions)
        phc = get_phc(eval_predictions)

        self.log(f"{sym_tag}_{step_tag}_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=batch.size)
        self.log(f"{sym_tag}_{step_tag}_MAP", map, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=batch.size)
        self.log(f"{sym_tag}_{step_tag}_PHC", phc, on_step=False, on_epoch=True,
                 prog_bar=False, sync_dist=True, batch_size=batch.size)

        return loss, map, phc

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        points = torch.stack(batch.get_points())
        points = torch.transpose(points, 1, 2).float()

        plane_predictions, axis_discrete_predictions, axis_continue_predictions = self.net.forward(points)
        loss = torch.tensor(0.0, device=points.device)

        if plane_predictions is not None:
            plane_loss, plane_map, plane_phc = self._process_prediction(
                batch, plane_predictions, batch.get_plane_syms(), self.plane_loss,
                "plane", "train"
            )
            loss += plane_loss

        if axis_discrete_predictions is not None:
            discrete_axis_loss, map_discrete_axis, phc_discrete_axis = self._process_prediction(
                batch, axis_discrete_predictions, batch.get_axis_discrete_syms(), self.discrete_rotational_loss,
                "d_axis", "train"
            )
            loss += discrete_axis_loss

        if axis_continue_predictions is not None:
            continue_axis_loss, map_continue_axis, phc_continue_axis = self._process_prediction(
                batch, axis_continue_predictions, batch.get_axis_continue_syms(), self.continue_rotational_loss,
                "c_axis", "train"

            )
            loss += continue_axis_loss

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        points = torch.stack(batch.get_points())
        points = torch.transpose(points, 1, 2).float()

        plane_predictions, axis_discrete_predictions, axis_continue_predictions = self.net.forward(points)
        loss = torch.tensor(0.0, device=points.device)

        if plane_predictions is not None:
            plane_loss, plane_map, plane_phc = self._process_prediction(
                batch, plane_predictions, batch.get_plane_syms(), self.plane_loss,
                "plane", "val"
            )
            loss += plane_loss

        if axis_discrete_predictions is not None:
            discrete_axis_loss, map_discrete_axis, phc_discrete_axis = self._process_prediction(
                batch, axis_discrete_predictions, batch.get_axis_discrete_syms(), self.discrete_rotational_loss,
                "d_axis", "val"
            )
            loss += discrete_axis_loss

        if axis_continue_predictions is not None:
            continue_axis_loss, map_continue_axis, phc_continue_axis = self._process_prediction(
                batch, axis_continue_predictions, batch.get_axis_continue_syms(), self.continue_rotational_loss,
                "c_axis", "val"

            )
            loss += continue_axis_loss

        return loss

    def test_step(self, batch, batch_idx):
        idxs, points, planar_syms, axis_continue_syms, axis_discrete_syms, transforms = batch
        points = torch.transpose(points, 1, 2).float()

        plane_predictions, axis_discrete_predictions, axis_continue_predictions = self.net.forward(points)
        plane_c_hats, matched_plane_pred = self.matcher.get_optimal_assignment(points, plane_predictions, planar_syms)
        axis_discrete_c_hat, matched_axis_discrete_pred = self.matcher.get_optimal_assignment(points,
                                                                                              axis_discrete_predictions,
                                                                                              axis_discrete_syms)
        axis_continue_c_hat, matched_axis_continue_pred = self.matcher.get_optimal_assignment(points,
                                                                                              axis_continue_predictions,
                                                                                              axis_continue_syms)

        bundled_plane_predictions = (batch, plane_predictions, plane_c_hats, matched_plane_pred)

        ##################
        # Plane
        ##################
        plane_loss = self.plane_loss(bundled_plane_predictions)

        prediction = [(batch, plane_predictions)]
        mean_avg_precision = get_mean_average_precision(prediction)
        phc = get_phc(prediction)

        unscaled_batch, unscaled_plane_predictions = reverse_transformation(batch, plane_predictions)

        unscaled_prediction = [(unscaled_batch, unscaled_plane_predictions)]
        unscaled_mean_avg_precision = get_mean_average_precision(unscaled_prediction)
        unscaled_phc = get_phc(unscaled_prediction)

        self.log("test_loss", plane_loss, on_step=False, on_epoch=True,
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
