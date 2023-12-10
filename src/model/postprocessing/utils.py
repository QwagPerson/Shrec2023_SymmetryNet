import torch


def reverse_transformation_aux(idx, points, y_true, y_pred, transform):
    y_pred = y_pred.clone()
    idx, points, y_true = transform.inverse_transform(idx.clone(), points.clone(), y_true.clone())
    _, _, y_pred[:, 0:6] = transform.inverse_transform(idx.clone(), points.clone(), y_pred[:, 0:6].clone())
    return idx, points, y_pred, y_true


def reverse_transformation(batch, y_pred):
    idxs, points, y_true, transforms = batch
    batch_size = len(y_true)

    unscaled_idxs = []
    unscaled_points = []
    unscaled_y_true = []
    unscaled_y_pred = []
    for i in range(batch_size):
        (curr_unscaled_idxs, curr_unscaled_points,
         curr_unscaled_y_pred, curr_unscaled_y_true) = reverse_transformation_aux(
            idxs[i], points[i], y_true[i], y_pred[i], transforms[i]
        )
        unscaled_idxs.append(curr_unscaled_idxs)
        unscaled_points.append(curr_unscaled_points)
        unscaled_y_true.append(curr_unscaled_y_true)
        unscaled_y_pred.append(curr_unscaled_y_pred)

    unscaled_idxs = torch.vstack(unscaled_idxs).view(batch_size)
    unscaled_points = torch.vstack(unscaled_points).view(batch_size, -1, 3)
    unscaled_y_pred = torch.vstack(unscaled_y_pred).view(batch_size, -1, 7)
    unscaled_batch = unscaled_idxs, unscaled_points, unscaled_y_true, transforms
    return unscaled_batch, unscaled_y_pred
