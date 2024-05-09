from typing import List

import torch

from src.dataset.SymDatasetItem import SymDatasetItem


class SymDatasetBatcher:
    def __init__(self, item_list: List[SymDatasetItem]):
        self.item_list = item_list
        self.size = len(item_list)
        self.device = item_list[0].points.device

    def get_points(self):
        return [item.points for item in self.item_list]

    def get_plane_syms(self):
        return [item.plane_symmetries for item in self.item_list]

    def get_axis_continue_syms(self):
        return [item.axis_continue_symmetries for item in self.item_list]

    def get_axis_discrete_syms(self):
        return [item.axis_discrete_symmetries for item in self.item_list]

    def get_shape_type_classification_labels(self, device="cpu"):
        return torch.stack([item.get_shape_type_classification_label(device) for item in self.item_list])
