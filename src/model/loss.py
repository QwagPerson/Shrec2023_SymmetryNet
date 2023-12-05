import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

from src.model.plane_utils import SymPlane, get_angle


def get_cost_matrix(curr_points, curr_y_pred, curr_y_true, eps=1e-8):
    """

    :param curr_points: N x 3
    :param curr_y_pred: M x 7
    :param curr_y_true: K x 6
    :param eps: eps to avoid dividing by 0
    :return: torch tensor M x K cost matrix
    """
    cost_matrix = torch.zeros((curr_y_true.shape[0], curr_y_pred.shape[0]))
    for i in range(curr_y_true.shape[0]):
        for j in range(curr_y_pred.shape[0]):
            true_plane = SymPlane.from_tensor(curr_y_true[i, :])
            pred_plane = SymPlane.from_tensor(curr_y_pred[j, :])
            reflected_by_pred = pred_plane.reflect_points(curr_points)
            reflected_by_true = true_plane.reflect_points(curr_points)

            # check dim
            diff = torch.linalg.norm(reflected_by_true - reflected_by_pred, dim=0).mean()

            cost_matrix[i, j] = 1 / (diff + eps)

    return cost_matrix


def create_one_hot(matched_planes_index, length, device="cpu"):
    one_hot = torch.zeros(length)
    for idx in matched_planes_index:
        one_hot[idx] = 1
    return one_hot.to(device)


def get_optimal_assignment(points, y_pred, y_true):
    """
    Solves the linear_sum_assignment problem for the planes predicted and
    the ground truth
    :param points: B x N x 3
    :param y_pred: B x N x M x 7
    :param y_true: B x K x 6
    :return: B x N x M c_hat boolean vector where true if the predicted plane was a match
             Adjusted_y_pred B x N x min(M, K) x 7 y_pred ordered by the matching just calculated.
    """
    batch_size = y_pred.shape[0]
    n = y_pred.shape[1]
    m = y_pred.shape[2]
    k = y_true.shape[1]
    adjusted_y_pred_list = []
    c_hats = []
    for b_idx in range(batch_size):
        for n_idx in range(n):
            curr_points = points[b_idx, :, :]
            curr_y_pred = y_pred[b_idx, n_idx, :, :].detach()
            curr_y_true = y_true[b_idx, :, :]
            cost_matrix = get_cost_matrix(curr_points, curr_y_pred, curr_y_true)

            row_idx, col_idx = linear_sum_assignment(cost_matrix, maximize=True)

            curr_c_hat = create_one_hot(col_idx, m, y_true.device)
            adjusted_y_pred = y_pred[b_idx, n_idx, col_idx, :]
            adjusted_y_pred_list.append(adjusted_y_pred)
            c_hats.append(curr_c_hat)
    c_hat = torch.stack(c_hats, dim=0).view(batch_size, n, m)
    new_y_pred = torch.stack(adjusted_y_pred_list, dim=0).view(batch_size, n, min(m, k), 7)
    return c_hat, new_y_pred


def get_confidence_loss(y_pred, c_hat):
    """

    :param y_pred: B x M x 7
    :param c_hat: B x M
    :return:
    """
    bceloss = nn.BCELoss()
    confidences_pred = y_pred[:, :, :, -1]  # B x M x 1
    return bceloss(confidences_pred, c_hat)


def get_angle_loss(y_pred, y_true):
    """

    :param y_pred: B x N x K x 6
    :param y_true: B x K x 6
    :return:
    """
    loss = torch.tensor([0.0], device=y_true.device)
    for i in range(y_pred.shape[1]):
        loss += get_angle(y_pred[:, i, :, 0:3], y_true[:, :, 0:3]).mean()
    return loss


def get_distance(y_pred, y_true):
    """

    :param y_pred: B x K x 6
    :param y_true: B x K x 6
    :return:
    """
    normals_true = y_true[:, :, 0:3]
    normals_pred = y_pred[:, :, 0:3]

    points_true = y_true[:, :, 3::]
    points_pred = y_pred[:, :, 3::]

    ds = - torch.einsum('bnd,bnd->bn', points_true, normals_true)

    distances = torch.abs(torch.einsum('bnd,bnd->bn', points_pred, normals_true) + ds)  # B x N

    return distances


def get_point_loss(y_pred, y_true):
    """

    :param y_pred: B x N x K x 6
    :param y_true: B x K x 6
    :return:
    """

    loss = torch.tensor([0.0], device=y_true.device)
    for i in range(y_pred.shape[1]):
        loss += get_distance(y_pred[:, i, :, :], y_true[:, :, :]).mean()
    return loss


def calculate_loss(
        batch,
        y_pred,
):
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
    idxs, points, y_true, transforms = batch

    # B x M x 1
    c_hat, adjusted_y_pred = get_optimal_assignment(points, y_pred, y_true)

    confidence_loss = get_confidence_loss(y_pred, c_hat)

    angle_loss = get_angle_loss(adjusted_y_pred[:, :, :, 0:6], y_true)

    point_loss = get_point_loss(adjusted_y_pred[:, :, :, 0:6], y_true)

    return confidence_loss + angle_loss + point_loss
