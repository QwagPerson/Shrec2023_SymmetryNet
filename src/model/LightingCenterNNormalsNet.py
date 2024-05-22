from typing import Callable, Union

import lightning
import torch

from src.metrics.MAP import get_mean_average_precision
from src.metrics.PHC import get_phc
from src.model.CenterNNormalsNet import CenterNNormalsNet
from src.model.losses.DiscreteRotationalSymmetryLoss import DiscreteRotationalSymmetryLoss
from src.model.losses.NormalLoss import NormalLoss
from src.model.losses.ConfidenceLoss import ConfidenceLoss
from src.model.losses.DistanceLoss import DistanceLoss
from src.model.losses.ReflectionSymmetryDistance import ReflectionSymmetryDistance
from src.model.losses.ReflectionSymmetryLoss import ReflectionSymmetryLoss
from src.model.losses.RotationalSymmetryDistance import RotationalSymmetryDistance
from src.model.losses.RotationalSymmetryLoss import RotationalSymmetryLoss
from src.model.matchers.SimpleMatcher import SimpleMatcher
from src.model.matchers.cost_matrix_methods import calculate_cost_matrix_normals


class LightingCenterNNormalsNet(lightning.LightningModule):
    def __init__(self,
                 amount_of_plane_normals_predicted: int = 32,
                 amount_of_axis_discrete_normals_predicted: int = 16,
                 amount_of_axis_continue_normals_predicted: int = 16,
                 plane_loss: Union[ReflectionSymmetryLoss, str] = "default",
                 discrete_rotational_loss: Union[DiscreteRotationalSymmetryLoss, str] = "default",
                 continue_rotational_loss: Union[RotationalSymmetryLoss, str] = "default",
                 w1: float = 1.0,
                 w2: float = 1.0,
                 w3: float = 1.0,
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
        self.matcher = SimpleMatcher(self.cost_matrix_method, self.device)
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

        if plane_loss == "default":
            self.plane_loss = ReflectionSymmetryLoss(
                confidence_weight=1.0, confidence_loss=ConfidenceLoss(),
                normal_weight=1.0, normal_loss=NormalLoss(),
                distance_weight=1.0, distance_loss=DistanceLoss(),
                reflection_symmetry_distance_weight=0.1,
                reflection_symmetry_distance=ReflectionSymmetryDistance()
            )
        else:
            self.plane_loss = plane_loss
        self.plane_loss_tag = [
            "confidence",
            "normal",
            "distance",
            "ref_sym_distance"
        ]

        if discrete_rotational_loss == "default":
            self.discrete_rotational_loss = DiscreteRotationalSymmetryLoss(
                confidence_weight=1.0, confidence_loss=ConfidenceLoss(),
                normal_weight=1.0, normal_loss=NormalLoss(),
                distance_weight=1.0, distance_loss=DistanceLoss(),
                angle_weight=1.0, angle_loss=DistanceLoss(),
                rotational_symmetry_distance_weight=0.1,
                rotational_symmetry_distance=RotationalSymmetryDistance()
            )
        else:
            self.discrete_rotational_loss = discrete_rotational_loss

        self.discrete_rotational_loss_tag = [
            "confidence",
            "normal",
            "distance",
            "angle",
            "rot_sym_distance"
        ]
        if continue_rotational_loss == "default":
            self.continue_rotational_loss = RotationalSymmetryLoss(
                confidence_weight=1.0, confidence_loss=ConfidenceLoss(),
                normal_weight=1.0, normal_loss=NormalLoss(),
                distance_weight=1.0, distance_loss=DistanceLoss(),
                rotational_symmetry_distance_weight=0.1,
                rotational_symmetry_distance=RotationalSymmetryDistance()
            )
        else:
            self.continue_rotational_loss = continue_rotational_loss
        self.continue_rotational_loss_tag = [
            "confidence",
            "normal",
            "distance",
            "rot_sym_distance"
        ]

        self.net = CenterNNormalsNet(
            amount_of_plane_normals_predicted,
            amount_of_axis_discrete_normals_predicted,
            amount_of_axis_continue_normals_predicted,
            use_bn=self.use_bn,
            normalize_normals=self.normalize_normals,
            print_losses=self.print_losses,
        )
        self.save_hyperparameters(ignore=["net", "plane_loss", "discrete_rotational_loss", "continue_rotational_loss"])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

    def _log(
            self, metric_val, metric_name, sym_tag, step_tag, batch_size,
            on_step=True, on_epoch=True, prog_bar=False, sync_dist=True
    ):
        self.log(f"{sym_tag}_{step_tag}_{metric_name}", metric_val, on_step=on_step, on_epoch=on_epoch,
                 prog_bar=prog_bar, batch_size=batch_size, sync_dist=sync_dist)

    def sort_prediction(self, pred):
        sorted_indices = pred[:, -1].sort(descending=True)
        return pred[sorted_indices[1]]

    def _process_prediction(self, batch, sym_pred, sym_true, loss_fun, sym_tag, step_tag, losses_tags):
        c_hat, match_pred, match_true, pred2true, true2pred = self.matcher.get_optimal_assignment(batch.get_points(),
                                                                                                  sym_pred, sym_true)
        bundled_predictions = (batch, sym_pred, c_hat, match_pred, match_true)
        loss, others = loss_fun(bundled_predictions)

        eval_predictions = [(batch.get_points(), sym_pred, sym_true)]
        map = get_mean_average_precision(eval_predictions)
        phc = get_phc(eval_predictions)

        if self.print_losses or True:
            torch.set_printoptions  (linewidth=200)
            torch.set_printoptions  (precision=3)
            torch.set_printoptions  (sci_mode=False)
            print(f"Points shape {batch.get_points()[0].shape}")
            # this is another nice paranormal phenomenon that I'll dig into tomorrow (always happens with train/egg_keplero/039942-egg_keplero-gaussian-sym.txt)
            if sym_true is not None and sym_pred is not None and sym_true[0] is not None and sym_pred[0] is not None:
                print(f"Y_true shape {len(sym_true)} - {sym_true[0].shape = }")
                print(f"Y_pred shape {len(sym_pred)} - {sym_pred[0].shape = }")

                print(f"Y_true:\n{sym_true[0]}\n")
                print(f"Y_pred (truncated):\n{self.sort_prediction(sym_pred[0])[:sym_true[0].shape[0]]}\n")
                print(f"Y_pred:\n{self.sort_prediction(sym_pred[0])}\n")
                print(f"Loss  : {loss}\n")
                print(f"Others: {others}\n")

            if batch.size > 1:
                for b_idx in range(batch.size):
                    y_true = sym_true[b_idx]
                    y_pred = sym_pred[b_idx]
                    curr_y_true = y_true
                    curr_y_pred = y_pred
    
                    print(f"{[b_idx]} Y_true\n{curr_y_true}")
                    print(f"{[b_idx]} Y_pred\n{curr_y_pred}")
                    #print(f"{[b_idx]} Loss: {losses[b_idx].item()}")
                    #print(f"{[b_idx]} Loss: {others[b_idx].item()}")



        """
        for i in range(batch.size):
            a_pred2true = pred2true[i]
            if a_pred2true is None:
                continue
            else:
                a_pred2true = torch.tensor(a_pred2true, device=self.device).float()
                for j in range(len(a_pred2true)):
                    self._log(a_pred2true[j], f"pred2true_batchmember_{i}_sym_{j}", sym_tag, step_tag, batch.size, sync_dist=False)

        for i in range(batch.size):
            idx = torch.tensor(batch.item_list[i].idx, device=self.device).float()
            self._log(idx, f"id_batch_member_{i}", sym_tag, step_tag, batch.size, sync_dist=False)
        """

        for idx in range(others.shape[0]):
            self._log(others[idx], f"loss_{losses_tags[idx]}", sym_tag, step_tag, batch.size)

        self._log(loss, "loss", sym_tag, step_tag, batch.size, prog_bar=True)
        self._log(map, "map", sym_tag, step_tag, batch.size, prog_bar=True)
        self._log(phc, "phc", sym_tag, step_tag, batch.size)

        return loss, map, phc

    def _step(self, batch, step_tag):
        batch.device = self.device
        self.matcher.device = self.device
        points = torch.stack(batch.get_points())
        points = torch.transpose(points, 1, 2).float()

        plane_predictions, axis_discrete_predictions, axis_continue_predictions = self.net.forward(points)
        loss = torch.tensor(0.0, device=points.device)

        if plane_predictions is not None:
            plane_loss, plane_map, plane_phc = self._process_prediction(
                batch, plane_predictions, batch.get_plane_syms(), self.plane_loss,
                "plane", step_tag, self.plane_loss_tag
            )
            loss += plane_loss * self.w1

        if axis_discrete_predictions is not None:
            discrete_axis_loss, map_discrete_axis, phc_discrete_axis = self._process_prediction(
                batch, axis_discrete_predictions, batch.get_axis_discrete_syms(), self.discrete_rotational_loss,
                "d_axis", step_tag, self.discrete_rotational_loss_tag
            )
            loss += discrete_axis_loss * self.w2

        if axis_continue_predictions is not None:
            continue_axis_loss, map_continue_axis, phc_continue_axis = self._process_prediction(
                batch, axis_continue_predictions, batch.get_axis_continue_syms(), self.continue_rotational_loss,
                "c_axis", step_tag, self.continue_rotational_loss_tag

            )
            loss += continue_axis_loss * self.w3

        return loss

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        batch.device = self.device
        self.matcher.device = self.device

        points = torch.stack(batch.get_points())
        points = torch.transpose(points, 1, 2).float()

        plane_predictions, axis_discrete_predictions, axis_continue_predictions = self.net.forward(points)

        return batch, plane_predictions, axis_discrete_predictions, axis_continue_predictions

    def on_after_backward(self):
        for name, param in self.net.named_parameters():
            if param.grad is not None:
                if param.grad.isnan().any():
                    print(f"{name} got nan!")
