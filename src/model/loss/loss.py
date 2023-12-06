import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from lapsolver import solve_dense
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
            # reflected_by_pred = pred_plane.reflect_points(curr_points)
            # reflected_by_true = true_plane.reflect_points(curr_points)

            # check dim
            # diff = torch.linalg.norm(reflected_by_true - reflected_by_pred, dim=0).mean()
            a, b = curr_y_true[i, 0:3].unsqueeze(0).unsqueeze(0), curr_y_pred[j, 0:3].unsqueeze(0).unsqueeze(0)
            cost_matrix[i, j] = 1 / (get_angle(a, b).item() + eps)

    return cost_matrix


def create_one_hot(matched_planes_index, length, device="cpu"):
    one_hot = torch.zeros(length)
    for idx in matched_planes_index:
        one_hot[idx] = 1
    return one_hot.to(device)


def get_optimal_assignment_lapsolver(points, y_pred, y_true):
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
            cost_matrix = - get_cost_matrix(curr_points, curr_y_pred, curr_y_true)

            row_idx, col_idx = solve_dense(cost_matrix)

            curr_c_hat = create_one_hot(col_idx, m, y_true.device)
            adjusted_y_pred = y_pred[b_idx, n_idx, col_idx, :]
            adjusted_y_pred_list.append(adjusted_y_pred)
            c_hats.append(curr_c_hat)
    c_hat = torch.stack(c_hats, dim=0).view(batch_size, n, m)
    new_y_pred = torch.stack(adjusted_y_pred_list, dim=0).view(batch_size, n, min(m, k), 7)
    return c_hat, new_y_pred


def get_optimal_assignment(points, y_pred, y_true):
    """
    Solves the linear_sum_assignment problem for the planes predicted and
    the ground truth
    :param points: B x N x 3
    :param y_pred: B x N x M x 7
    :param y_true: B x K x 6
    :return: B x N x M c_hat boolean vector where true if the predicted plane was a match
             Adjusted_y_pred B x N x k x 7 y_pred ordered by the matching just calculated.
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
    new_y_pred = torch.stack(adjusted_y_pred_list, dim=0).view(batch_size, n, k, 7)
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
    B, N, K, D = y_pred.shape
    y_true_repeated = y_true.repeat(1, N, 1)
    y_pred_view = y_pred.view(B, -1, D)

    normals_true_rep = y_true_repeated[:, :, 0:3]
    normals_pred = y_pred_view[:, :, 0:3]

    return get_angle(normals_true_rep, normals_pred).sum()


def get_point_loss(y_pred, y_true):
    """

    :param y_pred: B x N x K x 6
    :param y_true: B x K x 6
    :return: loss: 1 The loss asociated with the distance between
    all the N x K points
    """
    B, N, K, D = y_pred.shape
    y_true_repeated = y_true.repeat(1, N, 1)
    y_pred_view = y_pred.view(B, -1, D)

    normals_true_rep = y_true_repeated[:, :, 0:3]
    points_true_rep = y_true_repeated[:, :, 3::]

    points_pred = y_pred_view[:, :, 3::]

    ds = - torch.einsum('bnd,bnd->bn', normals_true_rep, points_true_rep)

    distances = torch.abs(torch.einsum('bnd,bnd->bn', points_pred, normals_true_rep) + ds)  # B x N

    return distances.sum()


def calculate_loss_aux(curr_points, curr_y_true, curr_y_pred):
    """

    :param curr_points: N x 3
    :param curr_y_true: K x 6
    :param curr_y_pred: N x M x 7
    :return:
    """
    # M x 7
    aggregated_y_pred = curr_y_pred.mean(dim=0)



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
    idxs, points, y_true, transforms = batch
    loss = torch.tensor([0.0], device=points.device)
    bs = points.shape[0]
    for b_idx in range(bs):
        curr_idx = idxs[b_idx]
        curr_points = points[b_idx, :, :]
        curr_y_true = y_true[b_idx, :, :]
        curr_y_pred = y_pred[b_idx, :, :, :]

        loss += calculate_loss_aux(curr_points, curr_y_true, curr_y_pred)

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

    # y_pred = y_pred.mean(dim=1) # B x M x 7

    # B x M x 1
    c_hat, adjusted_y_pred = get_optimal_assignment(points, y_pred, y_true)

    confidence_loss = get_confidence_loss(y_pred, c_hat)

    angle_loss = get_angle_loss(adjusted_y_pred[:, :, :, 0:6], y_true)

    point_loss = get_point_loss(adjusted_y_pred[:, :, :, 0:6], y_true)

    return confidence_loss + angle_loss + point_loss
