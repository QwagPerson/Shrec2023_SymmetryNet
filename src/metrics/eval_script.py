import torch

from src.metrics.utils import get_diagonals_length
from src.utils.plane import SymPlane


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


def get_match_sequence_plane_symmetry(points, y_pred, y_true, eps, theta, confidence_threshold):
    dist_threshold = get_diagonals_length(points) * eps

    y_pred = [SymPlane.from_tensor(y_pred[idx, 0:6], y_pred[idx, -1]) for idx in range(y_pred.shape[0])]
    y_pred = [x for x in y_pred if x.confidence > confidence_threshold]
    y_pred = sorted(y_pred, key=lambda x: x.confidence, reverse=True)

    m = len(y_pred)
    k = y_true.shape[0]
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


def calculate_metrics(match_sequence, groundtruth_total):
    sequence_length = match_sequence.shape[0]
    match_amount = match_sequence.sum().long().item()

    num_retrieved = 0
    relevant_retrieved = 0
    map_ = 0.0
    phc = 0.0

    pr_curve = torch.zeros((match_amount, 2), device=match_sequence.device)
    for idx in range(sequence_length):
        if match_sequence[idx] == 1:
            if num_retrieved == 0:
                phc = 1.0

            recall = (relevant_retrieved + 1) / groundtruth_total
            precision = (relevant_retrieved + 1) / (num_retrieved + 1)

            pr_curve[relevant_retrieved, 0] = recall
            pr_curve[relevant_retrieved, 1] = precision

            map_ = map_ + precision
            relevant_retrieved = relevant_retrieved + 1

        num_retrieved += 1

    if match_amount != 0:
        map_ = map_ / match_amount

    return map_, phc, pr_curve


def calculate_metrics_from_predictions(predictions, match_sequence_fun, eps, theta, confidence_threshold):
    """

    :param predictions: List of predictions.
                        predictions[i] = (batch_points[i], batch_y_pred, batch_y_true)
    :param eps:
    :param theta:
    :return:
    """

    maps = []
    phcs = []
    pr_curves = []

    for prediction in predictions:
        batch_points, batch_y_pred, batch_y_true = prediction
        batch_size = batch_points.shape[0]
        for idx in range(batch_size):
            points = batch_points[idx]
            y_pred = batch_y_pred[idx]
            y_true = batch_y_true[idx]
            matches = match_sequence_fun(points, y_pred, y_true, eps, theta, confidence_threshold)
            map_, phc, pr_curve = calculate_metrics(matches, y_true.shape[0])

            maps.append(map_)
            phcs.append(phc)
            pr_curves.append(pr_curve)

    total_map = torch.tensor(maps).mean()
    total_phc = torch.tensor(phcs).mean()

    interpolated_pr_curves = [interpolate_pr_curve(x) for x in pr_curves]
    total_pr_curve = torch.stack(interpolated_pr_curves).mean(dim=0)

    return total_map, total_phc, total_pr_curve


if __name__ == "__main__":
    bs = 12
    h = 10
    gt = 4
    predictions = [
        [
            torch.rand((bs, 10, 3)), torch.rand((bs, h, 7)), [torch.rand(gt, 6) for _ in range(bs)]
        ] for _ in range(10)
    ]
    end = calculate_metrics_from_predictions(predictions, get_match_sequence_plane_symmetry, 0.4, 0.02, 0.5)
    print(end)
