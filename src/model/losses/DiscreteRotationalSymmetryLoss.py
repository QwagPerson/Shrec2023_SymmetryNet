import torch
from torch import nn

from src.model.losses.ConfidenceLoss import ConfidenceLoss
from src.model.losses.DistanceLoss import DistanceLoss
from src.model.losses.NormalLoss import NormalLoss
from src.model.losses.RotationalSymmetryDistance import RotationalSymmetryDistance


class DiscreteRotationalSymmetryLoss(nn.Module):
    def __init__(
            self,
            confidence_weight: float, confidence_loss: ConfidenceLoss,
            normal_weight: float, normal_loss: NormalLoss,
            distance_weight: float, distance_loss: DistanceLoss,
            angle_weight: float, angle_loss: DistanceLoss,
            rotational_symmetry_distance_weight: float, rotational_symmetry_distance: RotationalSymmetryDistance
    ):
        super().__init__()

        self.confidence_weight = confidence_weight
        self.normal_weight = normal_weight
        self.distance_weight = distance_weight
        self.angle_weight = angle_weight
        self.rotational_symmetry_distance_weight = rotational_symmetry_distance_weight

        self.confidence_loss = confidence_loss
        self.normal_loss = normal_loss
        self.distance_loss = distance_loss
        self.angle_loss = angle_loss
        self.rotational_symmetry_distance = rotational_symmetry_distance

    def forward(self, bundled_discrete_rotational_predictions):
        batch, predictions, c_hats, matched_pred, matched_real = bundled_discrete_rotational_predictions

        batch_size = batch.size
        loss_matrix = torch.zeros((batch_size, 5), device=batch.device)

        for b_idx in range(batch_size):
            item = batch.get_item(b_idx)

            curr_points = item.points

            curr_y_true = matched_real[b_idx]
            curr_y_pred = matched_pred[b_idx]

            curr_conf_true = c_hats[b_idx]
            curr_conf_pred = predictions[b_idx, :, -1]
            conf_loss = self.confidence_weight * self.confidence_loss(curr_conf_pred, curr_conf_true)

            if curr_y_true is None:
                loss_matrix[b_idx, 0] = conf_loss
                continue

            curr_normal_true = curr_y_true[:, 0:3]
            curr_center_true = curr_y_true[:, 3:6]
            curr_angle_true = curr_y_true[:, 6]

            curr_normal_pred = curr_y_pred[:, 0:3]
            curr_center_pred = curr_y_pred[:, 3:6]
            curr_angle_pred = curr_y_pred[:, 6]

            normal_loss = self.normal_weight * self.normal_loss(curr_normal_pred, curr_normal_true)

            distance_loss = self.distance_weight * self.distance_loss(curr_center_pred, curr_center_true)

            angle_loss = self.angle_weight * self.angle_loss(curr_angle_pred, curr_angle_true)

            rotational_symmetry_distance = (self.rotational_symmetry_distance_weight *
                                            self.rotational_symmetry_distance(
                                                curr_points,
                                                curr_normal_pred, curr_normal_true,
                                                curr_center_pred, curr_normal_true,
                                                curr_angle_pred, curr_angle_true
                                            )
                                            )

            loss_matrix[b_idx, 0] = conf_loss
            loss_matrix[b_idx, 1] = normal_loss
            loss_matrix[b_idx, 2] = distance_loss
            loss_matrix[b_idx, 3] = rotational_symmetry_distance
            loss_matrix[b_idx, 4] = angle_loss

        loss = torch.sum(loss_matrix) / batch_size
        return loss, loss_matrix.sum(dim=0)
