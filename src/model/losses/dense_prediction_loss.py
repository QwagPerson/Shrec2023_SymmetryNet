import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

from src.model.losses.utils import get_optimal_assignment, calculate_angle_loss, calculate_distance_loss
from src.utils.plane import SymPlane


def calculate_loss_aux(curr_points, curr_y_pred, curr_y_true):
    """

    :param curr_points: N x 3
    :param curr_y_pred: N x M x 7
    :param curr_y_true: K x 6
    :return:
    """
    # M x 7
    aggregated_y_pred = curr_y_pred.mean(dim=0)
    # M
    confidences = aggregated_y_pred[:, -1]

    # c_hat : One-Hot M
    # matched_y_pred : K x 7
    c_hat, matched_y_pred = get_optimal_assignment(curr_points, aggregated_y_pred, curr_y_true)

    confidence_loss = nn.functional.binary_cross_entropy(confidences, c_hat)

    angle_loss = calculate_angle_loss(matched_y_pred[:, 0:6], curr_y_true)

    distance_loss = calculate_distance_loss(matched_y_pred[:, 0:6], curr_y_true)

    return confidence_loss + angle_loss + distance_loss


def calculate_loss(batch, y_pred):
    """

    :param batch: Tuple of idxs, points, sym_planes, transforms
        idxs : tensor of shape B,
        points : tensor of shape B x N x 3
        y_true : tensor of shape B x K x 6
        transforms : list of B elements where transform_i = Shrec2023Transform applied to element i of
        the previous three elements.
    :param y_pred: tensor fo shape B x N x M x 7
    :return:
    """
    _, points, y_true, transforms = batch
    loss = torch.tensor([0.0], device=points.device)
    bs = points.shape[0]
    for b_idx in range(bs):
        curr_points = points[b_idx, :, :]
        curr_y_true = y_true[b_idx]
        curr_y_pred = y_pred[b_idx]
        loss += calculate_loss_aux(curr_points, curr_y_pred, curr_y_true) / bs
    return loss


def calculate_loss_2(batch, y_pred):
    """

    :param batch: Tuple of idxs, points, sym_planes, transforms
        idxs : tensor of shape B,
        points : tensor of shape B x N x 3
        y_true : tensor of shape B x K x 6
        transforms : list of B elements where transform_i = Shrec2023Transform applied to element i of
        the previous three elements.
    :param y_pred: List of tuples of shape
                center_pred      -> B x N x 3
                normals_pred     -> B x N x M x 3
                confidences_pred -> B x N x M
    :return:
    """
    _, points, y_true, transforms = batch
    loss = torch.tensor([0.0], device=points.device)
    b, n, _ = points.shape
    for b_idx in range(b):
        curr_points = points[b_idx, :, :]
        curr_y_true = y_true[b_idx, :, :]
        center_pred, normal_pred, confidences_pred = y_pred[b_idx]
        k = normal_pred.shape[2]
        center_pred = center_pred.unsqueeze(dim=2).repeat(1, 1, k, 1)  # B x N x M x 3
        confidences_pred = confidences_pred.unsqueeze(dim=-1)  # B x N x M x 1
        curr_y_pred = torch.cat((normal_pred, center_pred, confidences_pred), dim=-1)
        loss += calculate_loss_aux(curr_points, curr_y_pred, curr_y_true) / b
    return loss


if __name__ == "__main__":
    idxs = torch.tensor([0])
    points = torch.rand((1, 100, 3))
    y_true = torch.rand((1, 1, 1, 7))
    y_true[:, :, :, -1] = 1.0
    y_pred = [
        (
            torch.rand((1, 100, 3)),
            torch.rand((1, 100, 2, 3)),
            torch.rand((1, 100, 2)),
        )
    ]
    y_true = y_true.squeeze(dim=1)
    y_true = y_true[:, :, 0:6]
    print(calculate_loss_2(
        (idxs, points, y_true, None),
        y_pred
    ))
