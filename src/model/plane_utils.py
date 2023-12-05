import math

import torch.linalg


def get_angle(a, b, radians=True):
    """

    :param a: Shape B x N x 3
    :param b: Shape B x N x 3
    :param radians: Flag to determine if return degrees or radians
    :return: Shape B x N of angles  between each vector
    """
    inner_product = torch.einsum('bnd,bnd->bn', a, b)
    a_norm = torch.linalg.norm(a, dim=2)
    b_norm = torch.linalg.norm(b, dim=2)
    # Avoiding div by 0
    cos = inner_product / ((a_norm * b_norm) + 1e-8)
    cos = torch.clamp(cos, -1, 1)
    angle = torch.acos(cos)
    if radians:
        return angle # * 180 / math.pi
    else:
        return angle * 180 / math.pi


class SymPlane:
    def __init__(self, normal, point):
        self.normal = torch.nn.functional.normalize(normal, dim=0)
        self.point = point
        self.offset = - torch.dot(point, normal)

    @staticmethod
    def from_tensor(plane_tensor):
        return SymPlane(plane_tensor[0:3], plane_tensor[3:6])

    def get_distance_to_points(self, points):
        return torch.einsum("nd, d -> n", points, self.normal) + self.offset

    def reflect_points(self, points):
        distances = self.get_distance_to_points(points)
        return points - 2 * torch.einsum('p,d->pd', distances,  self.normal)
