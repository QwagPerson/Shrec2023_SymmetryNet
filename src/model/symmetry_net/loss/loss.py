import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from src.model.plane import SymPlane


# from lapsolver import solve_dense

def calculate_cost_matrix_old(points, y_pred, y_true):
    """

    :param points: N x 3
    :param y_pred: K x 7
    :param y_true: M x 6
    :return:
    """
    cost_matrix = torch.zeros(y_pred.shape[0], y_true.shape[0])

    for i in range(y_pred.shape[0]):
        for j in range(y_true.shape[0]):
            pred_plane = SymPlane.from_tensor(y_pred[i, 0:6])
            true_plane = SymPlane.from_tensor(y_true[j, 0:6])
            cost_matrix[i, j] = pred_plane.get_angle_between_planes(true_plane)

    return cost_matrix


def calculate_cost_matrix(points, y_pred, y_true):
    """

    :param points: N x 3
    :param y_pred: K x 7
    :param y_true: M x 6
    :return: K x M
    """
    normals_pred = torch.nn.functional.normalize(y_pred[:, 0:3])
    normals_true = torch.nn.functional.normalize(y_true[:, 0:3])

    return torch.acos(normals_pred @ normals_true.T)


def calculate_cost_matrix_paper(points, y_pred, y_true):
    """

    :param points: N x 3
    :param y_pred: K x 7
    :param y_true: M x 6
    :return: K x M
    """
    cost_matrix = torch.zeros(y_pred.shape[0], y_true.shape[0])

    for i in range(y_pred.shape[0]):
        for j in range(y_true.shape[0]):
            pred_plane = SymPlane.from_tensor(y_pred[i, 0:6])
            true_plane = SymPlane.from_tensor(y_true[j, 0:6])
            cost_matrix[i, j] = torch.abs(
                torch.norm(
                    true_plane.reflect_points(points) - pred_plane.reflect_points(points),
                    dim=0
                )
            ).sum()

    return cost_matrix


def create_onehot(row_idx, length):
    """

    :param row_idx: Array of index of matches
    :param length: length of the vector
    :return:
    """
    out = torch.zeros(length)
    out[row_idx] = 1
    return out

    pass


def get_optimal_assignment(points, y_pred, y_true):
    """

    :param points: N x 3
    :param y_pred: M x 7
    :param y_true: K x 6
    :return:
    """
    m = y_pred.shape[0]
    cost_matrix = calculate_cost_matrix_paper(points, y_pred, y_true)
    row_id, col_id = linear_sum_assignment(cost_matrix.detach().numpy())
    c_hat = create_onehot(row_id, m)
    y_pred = y_pred[row_id, :]
    return c_hat, y_pred


def calculate_angle_loss(y_pred, y_true):
    """

    :param y_pred: M x 6
    :param y_true: M x 6
    :return:
    """
    normals_pred = torch.nn.functional.normalize(y_pred[:, 0:3], dim=1)  # M x 3
    normals_true = torch.nn.functional.normalize(y_true[:, 0:3], dim=1)  # M x 3

    angles = torch.acos(torch.clamp(normals_true @ normals_pred.T, -1, 1))
    return angles.min(dim=0).values.mean()


def calculate_distance_loss(y_pred, y_true):
    """

    :param y_pred: M x 6
    :param y_true: M x 6
    :return:
    """
    points_pred = y_pred[3:6]
    points_true = y_true[3:6]

    distances = torch.norm(points_true - points_pred, p=2, dim=0)

    return distances.mean()


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
    print(confidence_loss, angle_loss, distance_loss)
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
        curr_y_true = y_true[b_idx, :, :]
        curr_y_pred = y_pred[b_idx, :, :, :]
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
        confidences_pred = confidences_pred.unsqueeze(dim=-1) # B x N x M x 1
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
