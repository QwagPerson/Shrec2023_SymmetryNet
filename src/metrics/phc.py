import torch

from src.utils.plane import SymPlane


def get_diagonals_length(points: torch.Tensor):
    """
    :param points: Shape N x 3
    :return: length Shape 1
    """
    diagonal = points.max(dim=0).values - points.min(dim=0).values
    return torch.linalg.norm(diagonal)


def phc_match(points, y_pred, y_true, eps, theta):
    """

    :param points: N x 3
    :param y_pred: M x 7
    :param y_true: K x 6
    :param eps: float
    :param theta: float
    :return:
    """
    diag = get_diagonals_length(points)
    matched = False

    # Sort y_pred by confidence
    confidences = y_pred[:, -1].sort(descending=True).indices
    y_pred = y_pred[confidences][0]
    y_pred = SymPlane.from_tensor(y_pred, y_pred[-1])

    for idx in range(y_true.shape[0]):
        a_y_true = SymPlane.from_tensor(y_true[idx])
        match = y_pred.is_close(
            a_y_true, distance_threshold=eps * diag, angle_threshold=theta
        )

        matched = matched or match

    return matched


def calculate_phc(batch, y_pred_list, eps=0.01, theta=0.0174533):
    """
    :param batch:
        :param batched_points: B x N x 3
        :param y_true_list: List[B] -> K x 6
    :param y_pred_list: List[B] -> M x 7
    :param theta:
    :param eps:
    :return: float
    """
    _, batched_points, y_true_list, _ = batch
    b = len(y_true_list)
    good_matches = 0
    for idx in range(b):
        points = batched_points[idx]
        y_true = y_true_list[idx]
        y_pred = y_pred_list[idx]
        good_matches += phc_match(points, y_pred, y_true, eps, theta)
    return good_matches
