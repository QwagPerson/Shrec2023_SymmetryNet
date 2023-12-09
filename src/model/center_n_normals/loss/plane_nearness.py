import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

from src.utils.plane import SymPlane


def calculate_cost_matrix_normals(points, y_pred, y_true):
    """

    :param points: N x 3
    :param y_pred: K x 7
    :param y_true: M x 6
    :return: K x M
    """
    normals_pred = torch.nn.functional.normalize(y_pred[:, 0:3])
    normals_true = torch.nn.functional.normalize(y_true[:, 0:3])

    return torch.acos(normals_pred @ normals_true.T)


def calculate_cost_matrix_sde(points, y_pred, y_true):
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
            if cost_matrix[i, j].isnan().any():
                print("Got cost matrix nan with planes:")
                print("pred", pred_plane)
                print("true", true_plane)

    return cost_matrix


def create_onehot(row_idx, length, device="cpu"):
    """

    :param row_idx: Array of index of matches
    :param length: length of the vector
    :return:
    """
    out = torch.zeros(length, device=device)
    out[row_idx] = 1
    return out


def get_optimal_assignment(points, y_pred, y_true):
    """

    :param points: N x 3
    :param y_pred: M x 7
    :param y_true: K x 6
    :return:
    """
    m = y_pred.shape[0]
    cost_matrix = calculate_cost_matrix_sde(points, y_pred.detach().clone(), y_true)
    try:
        row_id, col_id = linear_sum_assignment(cost_matrix.detach().numpy())
    except Exception as e:
        print(e)
        print(cost_matrix.isnan().any())
    c_hat = create_onehot(row_id, m, device=points.device)
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

    # cos(theta) = n_1 . n_2.T
    # => if n_1 == n_2 => cos(theta) = 1
    # or if n_1 == -n_2 => cos(Theta) = -1
    # Min theta <=> Min 1 - |cos(Theta)| <=> 1 - |n_1 . n_2.T|
    cos_angle = 1 - torch.abs(normals_true @ normals_pred.T)
    return cos_angle.min(dim=0).values.mean()


def calculate_distance_loss(y_pred, y_true):
    """

    :param y_pred: M x 6
    :param y_true: M x 6
    :return:
    """
    points_pred = y_pred[:, 3:6]
    points_true = y_true[:, 3:6]

    distances = torch.norm(points_true - points_pred, p=1, dim=0)

    return distances.mean()


def calculate_loss_aux(points, y_pred, y_true):
    """

    :param points: N x 3
    :param y_pred: M x 7
    :param y_true: K x 6
    :return:
    """
    confidences = y_pred[:, -1]

    # c_hat : One-Hot M
    # matched_y_pred : K x 7
    c_hat, matched_y_pred = get_optimal_assignment(points, y_pred, y_true)

    confidence_loss = nn.functional.binary_cross_entropy(confidences, c_hat)

    angle_loss = calculate_angle_loss(matched_y_pred[:, 0:6], y_true)

    distance_loss = calculate_distance_loss(matched_y_pred[:, 0:6], y_true)
    loss = confidence_loss + angle_loss + distance_loss
    if y_pred.isnan().any():
        print("A head has nan")
        for i in range(y_pred.shape[0]):
            head_val = y_pred[i]
            if head_val.isnan().any():
                print(f"It is head number {i}!")

    if loss.isnan().any():
        print("conf_loss", confidence_loss, "angle_loss", angle_loss, "distance_loss", distance_loss)
    return loss


def calculate_loss(batch, y_pred):
    """
    :param batch: Tuple of idxs, points, sym_planes, transforms
        idxs : tensor of shape B
        points : tensor of shape B x N x 3
        y_true : List of B tensor of shape K x 6
        transforms : List of transforms used
    :param y_pred: tensor   B x H x 7
    :return:
    """
    _, points, y_true, _ = batch
    loss = torch.tensor([0.0], device=points.device)
    bs = points.shape[0]
    for b_idx in range(bs):
        curr_points = points[b_idx]
        curr_y_true = y_true[b_idx]
        curr_y_pred = y_pred[b_idx]
        loss += calculate_loss_aux(curr_points, curr_y_pred, curr_y_true)
    return loss / bs


if __name__ == "__main__":
    out = calculate_loss(
        (None, torch.ones(1, 10, 3), [torch.ones(2, 6)], None),
        torch.ones(1, 3, 7)
    )
    print(out)
