import torch
from torch import nn

from src.model.losses.ConfidenceLoss import ConfidenceLoss
from src.model.losses.DistanceLoss import DistanceLoss
from src.model.losses.NormalLoss import NormalLoss
from src.model.losses.ReflectionSymmetryDistance import ReflectionSymmetryDistance


class ReflectionSymmetryLoss(nn.Module):
    def __init__(
            self,
            confidence_weight: float, confidence_loss: ConfidenceLoss,
            normal_weight: float, normal_loss: NormalLoss,
            distance_weight: float, distance_loss: DistanceLoss,
            reflection_symmetry_distance_weight: float, reflection_symmetry_distance: ReflectionSymmetryDistance
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
            item = batch.get_item(b_idx)

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

        if True:
            total_loss = loss
            torch.set_printoptions  (linewidth=200)
            torch.set_printoptions  (precision=3)
            torch.set_printoptions  (sci_mode=False)
            print(f"\n")
            print(f"REF conf_loss    : {(conf_loss / total_loss).item():.2f} | {conf_loss.item()}")
            print(f"REF sde_loss     : {(reflection_symmetry_distance / total_loss).item():.2f} | {reflection_symmetry_distance.item()}")
            print(f"REF normal_loss  : {(normal_loss / total_loss).item():.2f} | {normal_loss.item()}")
            print(f"REF distance_loss: {(distance_loss / total_loss).item():.2f} | {distance_loss.item()}")
            print(f"REF Total_loss   : {total_loss.item():.2f}")
            print(f"\n")

        return loss, loss_matrix.sum(dim=0)
