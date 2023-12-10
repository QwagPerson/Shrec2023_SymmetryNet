import torch

from src.metrics.utils import get_diagonals_length
from src.utils.plane import SymPlane


def get_match_sequence(y_pred, y_true, points, eps, theta):
    m = y_pred.shape[0]
    k = y_true.shape[0]

    dist_threshold = get_diagonals_length(points) * eps

    y_pred = [SymPlane.from_tensor(y_pred[idx, 0:6], y_pred[idx, -1]) for idx in range(m)]
    y_pred = sorted(y_pred, key=lambda x: x.confidence, reverse=True)

    y_true = [SymPlane.from_tensor(y_true[idx, 0:6]) for idx in range(k)]
    match_sequence = torch.zeros(m)

    for pred_idx, pred_plane in enumerate(y_pred):
        match_idx = -1

        for true_idx, true_plane in enumerate(y_true):
            if pred_plane.is_close(true_plane, angle_threshold=theta, distance_threshold=dist_threshold):
                match_idx = true_idx
                break

        if match_idx != -1:
            match_sequence[pred_idx] = 1
            y_true.pop(match_idx)

    return match_sequence


def get_pr_curve(match_sequence, groundtruth_total):
    sequence_length = match_sequence.shape[0]
    num_retrieved = 0
    relevant_retrieved = 0
    pr_curve = torch.zeros(sequence_length, 2, device=match_sequence.device)
    for idx in range(sequence_length):
        num_retrieved += 1

        if match_sequence[idx] == 1:
            relevant_retrieved += 1

        precision = relevant_retrieved / num_retrieved
        recall = relevant_retrieved / groundtruth_total

        pr_curve[idx, 0] = recall
        pr_curve[idx, 1] = precision

    return pr_curve


def interpolate_pr_curve(uninterpolated_pr_curve, steps=11):
    recall_values = torch.linspace(0, 1, steps=steps, device=uninterpolated_pr_curve.device)
    precision_values = torch.zeros_like(recall_values)

    for idx in range(recall_values.shape[0]):
        current_recall = recall_values[idx]
        precision_at_current_recall = uninterpolated_pr_curve[
                                      current_recall <= uninterpolated_pr_curve[:, 0], :
                                      ]
        if precision_at_current_recall.shape[0] != 0:
            precision_values[idx] = precision_at_current_recall[:, 1].max()
        else:
            precision_values[idx] = 0

    return torch.vstack((recall_values, precision_values))


def calculate_area_under_curve(interpolated_pr_curve):
    return 1 / interpolated_pr_curve.shape[1] * interpolated_pr_curve[1, :].sum()


def calculate_average_precision(points, y_pred, y_true, eps, theta):
    """

    :param points:
    :param y_pred: M x 7
    :param y_true: K x 6
    :param eps:
    :param theta:
    :return:
    """
    match_sequence = get_match_sequence(y_pred, y_true, points, eps, theta)
    uninterpolated_pr_curve = get_pr_curve(match_sequence, y_true.shape[0])
    interpolated_pr_curve = interpolate_pr_curve(uninterpolated_pr_curve)
    average_precision = calculate_area_under_curve(interpolated_pr_curve)

    return average_precision


def get_average_precision(batch, y_pred_list, eps, theta):
    """

    :param batch:
        batched_points: B x N x 3
        y_true_list: List[B] -> K x 6
    :param y_pred_list: List[B] -> M x 7
    :param theta:
    :param eps:
    :return:
    """
    _, batched_points, y_true_list, _ = batch
    batch_size = len(y_true_list)
    average_precision_list = []
    for idx in range(batch_size):
        points = batched_points[idx]
        y_true = y_true_list[idx]
        y_pred = y_pred_list[idx]
        average_precision_list.append(
            calculate_average_precision(points, y_pred, y_true, eps, theta)
        )
    return average_precision_list


def get_mean_average_precision(predictions, eps=0.01, theta=0.0174533):
    """
    List of predictions
    :param predictions: List[P] where
        it contains the (batch, y_pred) that depends of batch_size
    :return:
    """
    device_used = predictions[0][1].device
    average_precisions_per_batch = []
    for (batch, y_pred) in predictions:
        average_precisions_per_batch.append(torch.tensor(get_average_precision(
            batch,
            y_pred,
            eps,
            theta
        ), device=device_used))
    average_precision_tensor = torch.hstack(average_precisions_per_batch).to(device_used)
    return average_precision_tensor.mean()
