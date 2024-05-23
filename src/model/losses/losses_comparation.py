# OLD LOSS IMPLEMENTAITON

import torch
from scipy.optimize import linear_sum_assignment

from src.dataset.SymDatasetBatcher import SymDatasetBatcher
from src.dataset.SymDatasetItem import SymDatasetItem
from src.model.losses.ConfidenceLoss import ConfidenceLoss
from src.model.losses.DistanceLoss import DistanceLoss
from src.model.losses.NormalLoss import NormalLoss
from src.model.losses.ReflectionSymmetryDistance import ReflectionSymmetryDistance
from src.model.losses.ReflectionSymmetryLoss import ReflectionSymmetryLoss
from src.model.matchers.SimpleMatcher import SimpleMatcher
from src.utils.plane import SymPlane


def calculate_cost_matrix_normals(points, y_pred, y_true):
    """
    :param points: N x 3
    :param y_pred: K x 7
    :param y_true: M x 6
    :return: K x M
    """
     normals_pred = torch.nn.functional.normalize(y_pred[:, 0:3], dim=1)
    normals_true = torch.nn.functional.normalize(y_true[:, 0:3], dim=1)

    return 1 - torch.abs(normals_pred @ normals_true.T)


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

    :param device:
    :param row_idx: Array of index of matches
    :param length: length of the vector
    :return:
    """
    out = torch.zeros(length, device=device)
    out[row_idx] = 1
    return out


def get_optimal_assignment(points, y_pred, y_true, method):
    """

    :param method:
    :param points: N x 3
    :param y_pred: M x 7
    :param y_true: K x 6
    :return:
    """
    m = y_pred.shape[0]
    cost_matrix = method(points, y_pred.detach().clone(), y_true)
    row_id, col_id = linear_sum_assignment(cost_matrix.cpu().detach().numpy())
    c_hat = create_onehot(row_id, m, device=points.device)
    y_pred = y_pred[row_id, :]
    y_true = y_true[col_id, :]
    return c_hat, y_pred, y_true


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


import torch
import torch.nn as nn

from src.utils.plane import SymPlane


def get_sde(points, pred_plane, true_plane, p=2):
    """
    :param points:
    :param pred_plane:
    print("true", true_plane)
    :param true_plane:
    :param p:
    :return:
    """
    pred_plane = SymPlane.from_tensor(pred_plane)
    true_plane = SymPlane.from_tensor(true_plane, normalize=True)

    return torch.norm(
        true_plane.reflect_points(points) - pred_plane.reflect_points(points),
        dim=1, p=p
    ).mean()


def calculate_sde_loss(points, y_pred, y_true):
    """

    :param points:
    :param y_pred: M x 6
    :param y_true: M x 6
    :return:
    """
    m = y_pred.shape[0]
    loss = torch.tensor([0.0], device=y_pred.device)
    for i in range(m):
        loss += get_sde(points, y_pred[i], y_true[i])
    return loss / m


def calculate_loss_aux(
        points,
        y_pred,
        y_true,
        cost_matrix_method,
        weights,
        show_loss_log=False
):
    """
    :param show_loss_log:
    :param weights:
    :param cost_matrix_method:
    :param points: N x 3
    :param y_pred: M x 7
    :param y_true: K x 6
    :return:
    """
    m = y_pred.shape[0]
    confidences = y_pred[:, -1]

    # c_hat : One-Hot M
    # matched_y_pred : K x 7
    c_hat, matched_y_pred, y_true = get_optimal_assignment(points, y_pred, y_true, cost_matrix_method)

    confidence_loss = nn.functional.binary_cross_entropy(confidences, c_hat) * weights[0]

    sde_loss = calculate_sde_loss(points, matched_y_pred[:, 0:6], y_true) * weights[1]

    distance_loss = calculate_distance_loss(matched_y_pred[:, 0:6], y_true) * weights[2]

    angle_loss = calculate_angle_loss(matched_y_pred[:, 0:6], y_true) * weights[3]

    total_loss = confidence_loss + sde_loss + angle_loss + distance_loss

    if show_loss_log or True:
        torch.set_printoptions(linewidth=200)
        torch.set_printoptions(precision=3)
        torch.set_printoptions(sci_mode=False)
        print(f"conf_loss    : {(confidence_loss / total_loss).item():.2f} | {confidence_loss.item()}")
        print(f"sde_loss     : {(sde_loss / total_loss).item():.2f} | {sde_loss.item()}")
        print(f"angle_loss   : {(angle_loss / total_loss).item():.2f} | {angle_loss.item()}")
        print(f"distance_loss: {(distance_loss / total_loss).item():.2f} | {distance_loss.item()}")
        print(f"Total_loss   : {total_loss.item():.2f}")

    return total_loss


def calculate_loss(
        batch,
        y_pred,
        cost_matrix_method=calculate_cost_matrix_normals,
        weights=torch.tensor([1.0, 1.0, 1.0, 1.0]),
        show_losses=False,
):
    """
    :param batch: Tuple of idxs, points, sym_planes, transforms
        idxs : tensor of shape B
        points : tensor of shape B x N x 3
        y_true : List of B tensor of shape K x 6
        transforms : List of transforms used
    :param y_pred: tensor   B x H x 7
    :param weights:
    :param cost_matrix_method:
    :return:
    """
    _, points, y_true, _ = batch
    bs = points.shape[0]
    loss = torch.tensor([0.0], device=points.device)
    losses = torch.zeros(bs, device=points.device)

    if show_losses:
        torch.set_printoptions(linewidth=200)
        torch.set_printoptions(precision=3)
        torch.set_printoptions(sci_mode=False)
        #print(f"Points shape {points.shape}")
        #print(f"Y_true shape {len(y_true)} - {y_true[0].shape = }")
        #print(f"Y_pred shape {len(y_pred)} - {y_pred.shape = }")

    for b_idx in range(bs):
        curr_points = points[b_idx]
        curr_y_true = y_true[b_idx]
        curr_y_pred = y_pred[b_idx]
        losses[b_idx] = calculate_loss_aux(
            curr_points, curr_y_pred, curr_y_true,
            cost_matrix_method, weights,
            show_losses
        )
        #if show_losses or losses[b_idx].item() >= 1. or curr_y_true.shape[0] >= 7:
            #print(f"{[b_idx]} Y_true\n{curr_y_true}")
            #print(f"{[b_idx]} Y_pred\n{curr_y_pred}")
            #print(f"{[b_idx]} Loss: {losses[b_idx].item()}")
    loss = torch.mean(losses)
    return loss  # / bs


if __name__ == "__main__":
    batch_size = 5
    n_points = 10
    n_heads = 5
    n_true_syms = 1

    random_points = torch.rand((batch_size, n_points, 3))
    random_y_pred = torch.rand((batch_size, n_heads, 7))
    random_y_true = [torch.rand((n_true_syms, 6)) for i in range(batch_size)]

    old_loss_result = calculate_loss(
        (None, random_points, random_y_true, None),
        random_y_pred,
        weights=torch.tensor([1.0, 0.1, 1.0, 1.0])
    )

    items = [SymDatasetItem("a-a-aa.xz", 0, random_points[i], random_y_true[i], None, None, None) for i in
             range(batch_size)]
    batch = SymDatasetBatcher(items)

    c_hat, match_pred, match_true, pred2true, true2pred = SimpleMatcher(method=calculate_cost_matrix_normals,
                                                                        device="cpu").get_optimal_assignment(
        batch.get_points(),
        random_y_pred, random_y_true)
    bundled_predictions = (batch, random_y_pred, c_hat, match_pred, match_true)
    plane_loss = ReflectionSymmetryLoss(
        confidence_weight=1.0, confidence_loss=ConfidenceLoss(weighted=False),
        normal_weight=1.0, normal_loss=NormalLoss(),
        distance_weight=1.0, distance_loss=DistanceLoss(),
        reflection_symmetry_distance_weight=0.1,
        reflection_symmetry_distance=ReflectionSymmetryDistance()
    )

    new_loss_result, others = plane_loss.forward(bundled_predictions)

    print("NEW", new_loss_result)
    print("OLD", old_loss_result)

    # === DEBUG ===
    print("other loss terms new")
    print(f"conf_loss    : {(others[0] / new_loss_result).item():.2f} | {others[0].item()}")
    print(f"sde_loss     : {(others[3] / new_loss_result).item():.2f} | {others[3].item()}")
    print(f"angle_loss   : {(others[1] / new_loss_result).item():.2f} | {others[1].item()}")
    print(f"distance_loss: {(others[2] / new_loss_result).item():.2f} | {others[2].item()}")
    print(f"Total_loss   : {new_loss_result.item():.2f}")
