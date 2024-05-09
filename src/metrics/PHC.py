from src.metrics.utils import get_diagonals_length
from src.utils.plane import SymPlane


def calculate_matching(points, y_pred, y_true, eps, theta):
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

    # Edge case where there are no known plane symmetries
    # Only true when the confidence is very low => The model knows there prob would be any symmetries.
    # That 0.1 should be a hparam but right now its fixed to test.
    if y_true is None:
        if y_pred.confidence < 0.1:
            return 1
        else:
            return 0

    for idx in range(y_true.shape[0]):
        a_y_true = SymPlane.from_tensor(y_true[idx])
        match = y_pred.is_close(
            a_y_true, distance_threshold=eps * diag, angle_threshold=theta
        )

        matched = matched or match

    return matched


def get_matches_amount(points_list, y_pred_list, y_true_list, eps, theta):
    batch_size = len(y_true_list)
    matches = 0
    for idx in range(batch_size):
        points = points_list[idx]
        y_true = y_true_list[idx]
        y_pred = y_pred_list[idx]
        matches += calculate_matching(points, y_pred, y_true, eps, theta)
    return matches / batch_size


def get_phc(predictions, eps=0.01, theta=0.0174533):
    total_matches = 0.0
    for (points_list, y_pred_list, y_true_list) in predictions:
        total_matches += get_matches_amount(
            points_list,
            y_pred_list,
            y_true_list,
            eps,
            theta
        )
    return total_matches / len(predictions)
