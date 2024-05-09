import torch
from torch import nn


class ReflectionSymmetryLoss(nn.Module):
    def __init__(
            self,
            confidence_weight, confidence_loss,
            normal_weight, normal_loss,
            distance_weight, distance_loss,
            reflection_symmetry_distance_weight, reflection_symmetry_distance
    ):
        super().__init__()

        self.confidence_weight = confidence_weight
        self.normal_weight = normal_weight
        self.distance_weight = distance_weight
        self.reflection_symmetry_distance_weight = reflection_symmetry_distance_weight

        self.confidence_loss = confidence_loss
        self.normal_loss = normal_loss
        self.distance_loss = distance_loss
        self.reflection_symmetry_distance = reflection_symmetry_distance

    def forward(self, bundled_plane_predictions):
        batch, plane_predictions, plane_c_hats, matched_plane_pred, matched_plane_real = bundled_plane_predictions

        batch_size = batch.size
        loss_matrix = torch.zeros((batch_size, 4), device=batch.device)

        for b_idx in range(batch_size):
            item = batch.item_list[b_idx]

            curr_points = item.points

            curr_y_true = matched_plane_real[b_idx]
            curr_y_pred = matched_plane_pred[b_idx]

            curr_conf_true = plane_c_hats[b_idx]
            curr_conf_pred = plane_predictions[b_idx, :, -1]
            conf_loss = self.confidence_weight * self.confidence_loss(curr_conf_pred, curr_conf_true)

            if curr_y_true is None:
                loss_matrix[b_idx, 0] = conf_loss
                continue

            curr_normal_true = curr_y_true[:, 0:3]
            curr_center_true = curr_y_true[:, 3:6]

            curr_normal_pred = curr_y_pred[:, 0:3]
            curr_center_pred = curr_y_pred[:, 3:6]

            normal_loss = self.normal_weight * self.normal_loss(curr_normal_pred, curr_normal_true)

            distance_loss = self.distance_weight * self.distance_loss(curr_center_pred, curr_center_true)

            reflection_symmetry_distance = (self.reflection_symmetry_distance_weight *
                                            self.reflection_symmetry_distance(
                                                curr_points,
                                                curr_normal_pred, curr_normal_true,
                                                curr_center_pred, curr_normal_true
                                            )
                                            )

            loss_matrix[b_idx, 0] = conf_loss
            loss_matrix[b_idx, 1] = normal_loss
            loss_matrix[b_idx, 2] = distance_loss
            loss_matrix[b_idx, 3] = reflection_symmetry_distance

        loss = torch.sum(loss_matrix) / batch_size
        return loss, loss_matrix.sum(dim=0)
