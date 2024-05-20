import torch
import torch.nn as nn

from src.model.losses.utils import get_optimal_assignment, calculate_distance_loss, calculate_angle_loss, \
    calculate_cost_matrix_normals
from src.utils.plane import SymPlane


def get_sde(points, pred_plane, true_plane, p=2, show_loss_log=False):
    """ 
    :param points:
    :param pred_plane:
    :param true_plane:
    :param p:
    :return:
    """
    if show_loss_log:
        print(f'pred_plane\n{pred_plane}')
        print(f'true_plane\n{true_plane}')
    pred_plane = SymPlane.from_tensor(pred_plane)
    true_plane = SymPlane.from_tensor(true_plane, normalize=True)
    if show_loss_log:
        print(f'pred_plane\n{pred_plane}')
        print(f'true_plane\n{true_plane}')
    ppr = pred_plane.reflect_points(points.T)
    tpr = true_plane.reflect_points(points.T)
    if show_loss_log:
        print(f'{len(ppr) = }')
        print(f'{len(tpr) = }')
        print(f'ppr\n{ppr}')
        print(f'tpr\n{tpr}')
    loss = torch.norm(tpr - ppr, dim=0, p=p).mean()
    if show_loss_log:
        print(f'SDE loss: {loss}')
    return loss


def get_sde2(points, pred_plane, true_plane, p=2):
    """
    :param points:
    :param pred_plane:
    :param true_plane:
    :param p:
    :return:
    """
    pred_plane = SymPlane.from_tensor(pred_plane)
    true_plane = SymPlane.from_tensor(true_plane, normalize=True)
    return torch.norm(
        true_plane.reflect_points(points) - pred_plane.reflect_points(points),
        dim=0, p=p
    ).mean()


def calculate_sde_loss(points, y_pred, y_true, show_loss_log=False):
    """

    :param points:
    :param y_pred: M x 6
    :param y_true: M x 6
    :return:
    """
    m = y_pred.shape[0]
    loss = torch.tensor([0.0], device=y_pred.device)
    for i in range(m):
        loss += get_sde(points, y_pred[i], y_true[i], show_loss_log=show_loss_log)
    return loss / m

def calculate_mae_loss(y_pred, y_true, show_loss_log=False):
    if show_loss_log:
        print(f"y_pred.shape: {y_pred.shape} - y_true.shape: {y_true.shape}")
        print(f"y_true: {y_true}")
        print(f"y_pred: {y_pred}")
    maeloss = torch.nn.L1Loss(reduction='none')
    loss    = maeloss(y_pred, y_true)
    if show_loss_log:
        print(f"Unreduced MAE Loss: {loss.sum().item()}")
        print(f"          MAE Loss: {loss.mean().item()}")
    return loss.sum() #.mean()

def get_dumb_assignment(points, y_pred, y_true, show_loss_log=False):
	if show_loss_log:
		print(f'{y_pred.shape = }')		# [27, 7]
		print(f'{y_true.shape = }')		# [5,  6]
	c_hat = torch.zeros([y_pred.shape[0]], device=points.device)
	ones  = torch.ones(y_true.shape[0], device=points.device)
	c_hat[:y_true.shape[0]] = ones
	matched_pred = y_pred[:y_true.shape[0], :]
	if show_loss_log:
		print(f'{c_hat.shape = }')
		print(f'{matched_pred.shape = }')
		print(f'{c_hat = }')
		print(f'{matched_pred = }')
	return c_hat, matched_pred

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
    c_hat, matched_y_pred = get_optimal_assignment(points, y_pred, y_true, cost_matrix_method)
    #c_hat, matched_y_pred = get_dumb_assignment(points, y_pred, y_true, show_loss_log=show_loss_log)

    confidence_loss = nn.functional.binary_cross_entropy(confidences, c_hat) * weights[0]

    sde_loss = calculate_sde_loss(points, matched_y_pred[:, 0:6], y_true, show_loss_log=show_loss_log) * weights[1]

    distance_loss = calculate_distance_loss(matched_y_pred[:, 0:6], y_true) * weights[2]

    angle_loss = calculate_angle_loss(matched_y_pred[:, 0:6], y_true) * weights[3]

    #mae_loss = calculate_mae_loss(matched_y_pred[:, 0:6], y_true, show_loss_log=show_loss_log)

    total_loss = confidence_loss + sde_loss + angle_loss + distance_loss # + mae_loss

    if show_loss_log:
        torch.set_printoptions  (linewidth=200)
        torch.set_printoptions  (precision=3)
        torch.set_printoptions  (sci_mode=False)
        print(f"Total_loss   : {total_loss.item():.2f}")
        print(f"Conf_loss    : {(confidence_loss / total_loss).item():.2f} | {confidence_loss.item()}")
        print(f"sde_loss     : {(sde_loss / total_loss).item():.2f} | {sde_loss.item()}")
        print(f"angle_loss   : {(angle_loss / total_loss).item():.2f} | {angle_loss.item()}")
        print(f"distance_loss: {(distance_loss / total_loss).item():.2f} | {distance_loss.item()}")
        #print(f"mae_loss     : {(mae_loss / total_loss).item():.2f} | {mae_loss.item()}")

    return total_loss


def calculate_loss(
        batch,
        y_pred,
        cost_matrix_method=calculate_cost_matrix_normals,
        weights=torch.tensor([1.0, 0.1, 1.0, 1.0]),
        show_losses=True,
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
    bs     = points.shape[0]
    loss   = torch.tensor([0.0], device=points.device)
    losses = torch.zeros(bs, device=points.device)

    if show_losses:
        torch.set_printoptions  (linewidth=200)
        torch.set_printoptions  (precision=3)
        torch.set_printoptions  (sci_mode=False)
        print(f"OLD Points shape {points.shape}")
        print(f"OLD Y_true shape {len(y_true)} - {y_true[0].shape}")
        print(f"OLD Y_pred shape {y_pred.shape}")

    for b_idx in range(bs):
        curr_points = points[b_idx]
        curr_y_true = y_true[b_idx]
        curr_y_pred = y_pred[b_idx]
        losses[b_idx] = calculate_loss_aux(
            curr_points, curr_y_pred, curr_y_true,
            cost_matrix_method, weights,
            show_losses)
        if show_losses:
            print(f"OLD {[b_idx]} Points: {curr_points}")
            print(f"OLD {[b_idx]} Y_true: {curr_y_true}")
            print(f"OLD {[b_idx]} Y_pred: {curr_y_pred}")
            print(f"OLD {[b_idx]} Loss  : {losses[b_idx].item()}")
    #final_loss = loss / bs
    loss = torch.mean(losses)
    #loss = torch.sum(losses)
    if show_losses:
        print(f"OLD Final loss: {loss.item()}")
        #print(f"Final loss: {final_loss.item()}")
    return loss
    #return final_loss
