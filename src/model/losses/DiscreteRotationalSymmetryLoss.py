import torch
from torch import nn


class DiscreteRotationalSymmetryLoss(nn.Module):
    def __init__(
            self,
            confidence_weight, confidence_loss,
            angle_weight, angle_loss,
            distance_weight, distance_loss,
            rotational_symmetry_distance_weight, rotational_symmetry_distance
    ):
        super().__init__()

        self.confidence_weight = confidence_weight
        self.angle_weight = angle_weight
        self.distance_weight = distance_weight
        self.rotational_symmetry_distance_weight = rotational_symmetry_distance_weight

        self.confidence_loss = confidence_loss
        self.angle_loss = angle_loss
        self.distance_loss = distance_loss
        self.rotational_symmetry_distance = rotational_symmetry_distance

    def forward(self, bundled_plane_predictions):
        batch, plane_predictions, plane_c_hats, matched_plane_pred, matched_plane_real = bundled_plane_predictions
        idxs, points, planar_syms, axis_continue_syms, axis_discrete_syms, transforms = batch

        batch_size = points.shape[0]
        losses = torch.zeros(batch_size, device=points.device)

        for b_idx in range(batch_size):

            curr_points = points[b_idx]

            curr_y_true = matched_plane_real[b_idx]
            curr_y_pred = matched_plane_pred[b_idx]

            curr_conf_true = plane_c_hats[b_idx]
            curr_conf_pred = plane_predictions[b_idx, :, -1]
            conf_loss = self.confidence_weight * self.confidence_loss(curr_conf_pred, curr_conf_true)

            if curr_y_true is None:
                losses[b_idx] = conf_loss
                continue

            curr_normal_true = curr_y_true[:, 0:3]
            curr_center_true = curr_y_true[:, 3:6]

            curr_normal_pred = curr_y_pred[:, 0:3]
            curr_center_pred = curr_y_pred[:, 3:6]

            angle_loss = self.angle_weight * self.angle_loss(curr_normal_pred, curr_normal_true)

            distance_loss = self.distance_weight * self.distance_loss(curr_center_pred, curr_center_true)

            reflection_symmetry_distance = (self.reflection_symmetry_distance_weight *
                                            self.reflection_symmetry_distance(
                                                curr_points,
                                                curr_normal_pred, curr_normal_true,
                                                curr_center_pred, curr_normal_true
                                            )
                                            )

            losses[b_idx] = (
                    conf_loss +
                    angle_loss +
                    distance_loss +
                    reflection_symmetry_distance
            )

        loss = torch.sum(losses) / batch_size
        return loss
