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
    def __init__(self, method, device):
        self.method = method
        self.device = device



    def get_fixed_assignment(self, points, y_pred, y_true):
        """
        :param points: N x 3
        :param y_pred: M x 7
        :param y_true: K x 6
        :return:
        """

        print(f'----------------------')
        print(f'----------------------')
        print(f'get_fixed_assignment()')
        print(f'----------------------')
        print(f'----------------------')

        print(f'y_pred orig: {y_pred}\ny_true orig: {y_true}')

        m = y_pred.shape[0]
        ngt = y_true.shape[0]
        #cost_matrix = self.method(points, y_pred.detach().clone(), y_true)
        #row_id, col_id = linear_sum_assignment(cost_matrix.cpu().detach().numpy())
        row_id, col_id = list(range(ngt)), list(range(ngt))

        print(f'm: {m}, row_id: {row_id}, col_id: {col_id}')
        #print(f'cost_matrix: {cost_matrix}')

        c_hat = create_onehot(row_id, m, device=self.device)

        print(f'c_hat: {c_hat}')

        y_pred = y_pred[row_id, :]
        y_true = y_true[col_id, :]

        print(f'y_pred: {y_pred}\ny_true: {y_true}')

        return c_hat, y_pred, y_true, row_id, col_id
    def get_optimal_assignment_mse(self, points, y_pred, y_true):
        """
        :param points: N x 3
        :param y_pred: M x 7
        :param y_true: K x 6
        :return:
        """

        print(f'----------------------------')
        print(f'----------------------------')
        print(f'get_optimal_assignment_mse()')
        print(f'----------------------------')
        print(f'----------------------------')

        print(f'y_pred orig: {y_pred}\ny_true orig: {y_true}')

        m = y_pred.shape[0]
        cost_matrix = self.method(points, y_pred.detach().clone(), y_true)
        row_id, col_id = linear_sum_assignment(cost_matrix.cpu().detach().numpy())

        print(f'm: {m}, row_id: {row_id}, col_id: {col_id}')
        print(f'cost_matrix: {cost_matrix}')

        c_hat = create_onehot(row_id, m, device=self.device)

        print(f'c_hat: {c_hat}')

        y_pred = y_pred[row_id, :]
        y_true = y_true[col_id, :]

        print(f'y_pred: {y_pred}\ny_true: {y_true}')

        return c_hat, y_pred, y_true, row_id, col_id

    def get_optimal_assignment_aux(self, points, y_pred, y_true):
        """
        :param points: N x 3
        :param y_pred: M x 7
        :param y_true: K x 6
        :return:
        """
        print(f'y_pred orig: {y_pred}\ny_true orig: {y_true}')

        m = y_pred.shape[0]
        cost_matrix = self.method(points, y_pred.detach().clone(), y_true)
        row_id, col_id = linear_sum_assignment(cost_matrix.cpu().detach().numpy())

        print(f'm: {m}, row_id: {row_id}, col_id: {col_id}')
        print(f'cost_matrix: {cost_matrix}')

        c_hat = create_onehot(row_id, m, device=self.device)

        print(f'c_hat: {c_hat}')

        y_pred = y_pred[row_id, :]
        y_true = y_true[col_id, :]

        print(f'y_pred: {y_pred}\ny_true: {y_true}')

        return c_hat, y_pred, y_true, row_id, col_id

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
        assignments_y_pred_2_y_true = []
        assignments_y_true_2_y_pred = []

        for idx in range(batch_size):
            if y_true[idx] is None:
                c_hats.append(torch.zeros(head_amount, device=self.device))
                matches_y_pred.append(None)
                matches_y_true.append(None)
                assignments_y_pred_2_y_true.append(None)
                assignments_y_true_2_y_pred.append(None)
            else:
                #c_hat, match_y_pred, match_y_true, pred2true, true2pred = self.get_optimal_assignment_aux(points[idx], y_pred[idx], y_true[idx])
                #c_hat, match_y_pred, match_y_true, pred2true, true2pred = self.get_optimal_assignment_mse(points[idx], y_pred[idx], y_true[idx])
                c_hat, match_y_pred, match_y_true, pred2true, true2pred = self.get_fixed_assignment(points[idx], y_pred[idx], y_true[idx])
                c_hats.append(c_hat)
                matches_y_pred.append(match_y_pred)
                matches_y_true.append(match_y_true)
                assignments_y_pred_2_y_true.append(pred2true)
                assignments_y_true_2_y_pred.append(true2pred)


        return c_hats, matches_y_pred, matches_y_true, assignments_y_pred_2_y_true, assignments_y_true_2_y_pred

