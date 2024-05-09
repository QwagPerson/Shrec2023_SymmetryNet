import torch
from scipy.optimize import linear_sum_assignment


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


class SimpleMatcher:
    def __init__(self, method):
        self.method = method



    def get_optimal_assignment_aux(self, points, y_pred, y_true):
        """

        :param points: N x 3
        :param y_pred: M x 7
        :param y_true: K x 6
        :return:
        """
        m = y_pred.shape[0]
        cost_matrix = self.method(points, y_pred.detach().clone(), y_true)
        row_id, col_id = linear_sum_assignment(cost_matrix.cpu().detach().numpy())
        c_hat = create_onehot(row_id, m, device=points.device)
        y_pred = y_pred[row_id, :]
        y_true = y_true[col_id, :]
        return c_hat, y_pred, y_true

    def get_optimal_assignment(self, points, y_pred, y_true):
        """

        :param points: B X N x 3
        :param y_pred: B X M x 7
        :param y_true: B X K x 6
        :return:
        """
        batch_size = y_pred.shape[0]
        head_amount = y_pred.shape[1]
        c_hats = []
        matches_y_pred = []
        matches_y_true = []

        for idx in range(batch_size):
            if y_true[idx] is None:
                c_hats.append(torch.zeros(head_amount))
                matches_y_pred.append(None)
                matches_y_true.append(None)
            else:
                c_hat, match_y_pred, match_y_true = self.get_optimal_assignment_aux(points[idx], y_pred[idx], y_true[idx])
                c_hats.append(c_hat)
                matches_y_pred.append(match_y_pred)
                matches_y_true.append(match_y_true)


        return c_hats, matches_y_pred, matches_y_true

