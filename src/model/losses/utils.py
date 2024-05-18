import torch
from scipy.optimize import linear_sum_assignment

from src.utils.plane import SymPlane



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
    cos_angle = 1 - torch.abs(normals_true @ normals_pred.T)  # M x M
    # We take the min of cos_angle because it is expected that the min value is
    # the one corresponding to the assigned plane, this is not a 100% certain but
    # it is expected because this plane was assigned to that true plane because of a reason
    # anyways this is fast but unreadable as #!=$@ idk how i would put it better tho
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
