from sklearn.cluster import DBSCAN
import torch

from src.model.plane import SymPlane


def distance_between_planes_fn(x1, x2):
    x1, x2 = x1[0:6], x2[0:6]
    x1 = SymPlane.from_array(x1)
    x2 = SymPlane.from_array(x2)
    angles, distance = x1.get_distances(x2)
    out = angles
    out = out.item()
    return out


def nms(predictions):
    """

    :param predictions: M x 7
    :return:
    """
    keep = []
    confidences_sorted_values, confidences_sorted_idx = torch.sort(predictions[:, -1], descending=True)
    predictions = predictions[confidences_sorted_idx]
    predictions = [
        SymPlane.from_tensor(
            predictions[idx, 0:6], predictions[idx, -1].item()
        ) for idx in range(predictions.shape[0])
    ]
    #is_close_matrix

    while len(predictions) > 0:
        best_plane = predictions.pop(0)
        keep.append(best_plane)
        remove_list = []
        for idx, another_plane in enumerate(predictions):
            if best_plane.is_close(another_plane):
                remove_list.append(idx)

        remove_list = sorted(list(set(remove_list)))
        for idx in reversed(remove_list):
            predictions.pop(idx)

    keep = [x.to_tensor() for x in keep]
    if len(keep) > 0:
        keep = torch.vstack(keep)
    else:
        keep = torch.tensor([])
    return keep


def postprocess_predictions(y_pred, eps=0.2, min_samples=500, n_jobs=1):
    """

    :param y_pred: B x N x M x 7
    :return:
    """
    bs, n, m, _ = y_pred.shape
    out_predictions = []
    # Para cada batch
    for i in range(bs):
        x = y_pred[i, :, :, 0:3].view(n * m, -1)  # nm x 3
        x = torch.nn.functional.normalize(x, dim=1)
        # Handle edge cases where x = 1 but torch.acos returns nan because
        # the representation of the number is bigger than one strangely
        x = torch.clamp(x @ x.T, -1, 1)
        # MN x 3 @ 3 x MT -> MN x MN where each cell is the distance
        # because is the dot product
        distances = torch.acos(x)
        weights = y_pred[i, :, :, -1].view(n * m)
        # Entrenamos DBSCAN sobre todos los planos predichos
        # Los puntos core seran los nuevos planos
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed", n_jobs=n_jobs)
        dbscan = dbscan.fit(distances, sample_weight=weights)
        core_vectors = x[dbscan.core_sample_indices_, :]
        out = nms(core_vectors)
        out_predictions.append(out)

    return out_predictions


if __name__ == "__main__":
    mock_y_pred = torch.rand(3, 10, 1, 7)
    preds = postprocess_predictions(mock_y_pred, 1, 2, n_jobs=4)
    print(preds)

