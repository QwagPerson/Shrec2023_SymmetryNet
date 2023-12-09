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

    for idx in range(y_true.shape[0]):
        a_y_true = SymPlane.from_tensor(y_true[idx])
        match = y_pred.is_close(
            a_y_true, distance_threshold=eps * diag, angle_threshold=theta
        )

        matched = matched or match

    return matched


def get_matches_amount(batch, y_pred_list, eps, theta):
    """
    :param batch:
        batched_points: B x N x 3
        y_true_list: List[B] -> K x 6
    :param y_pred_list: List[B] -> M x 7
    :param theta:
    :param eps:
    :return: float
    """
    _, batched_points, y_true_list, _ = batch
    batch_size = len(y_true_list)
    matches = 0
    for idx in range(batch_size):
        points = batched_points[idx]
        y_true = y_true_list[idx]
        y_pred = y_pred_list[idx]
        matches += calculate_matching(points, y_pred, y_true, eps, theta)
    return matches / batch_size


def get_phc(predictions, eps=0.01, theta=0.0174533):
    """
    List of predictions
    :param eps: Percentage of diagonal. Controls distance Threshold
    :param theta: Percentage of Angle. Control angle Threshold
    :param predictions: List[P] where
        it contains the (batch, y_pred) that depends of batch_size
    :return:
    """
    total_matches = 0.0
    for (batch, y_pred) in predictions:
        total_matches += get_matches_amount(
            batch,
            y_pred,
            eps,
            theta
        )
    return total_matches / len(predictions)
