from typing import Optional
import torch

from src.dataset.transforms.AbstractTransform import AbstractTransform


class ReverseTransform(AbstractTransform):
    def __init__(self, a_transform):
        self.a_transform = a_transform

    def inverse_transform(
            self,
            idx: int,
            points: torch.Tensor,
            planar_symmetries: Optional[torch.Tensor],
            axis_continue_symmetries: Optional[torch.Tensor],
            axis_discrete_symmetries: Optional[torch.Tensor]
    ) -> (int, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]):
        (idx, points, planar_symmetries,
         axis_continue_symmetries, axis_discrete_symmetries) = self.a_transform.transform(
            idx, points, planar_symmetries, axis_continue_symmetries, axis_discrete_symmetries
        )
        return idx, points, planar_symmetries, axis_continue_symmetries, axis_discrete_symmetries

    def transform(
            self,
            idx: int,
            points: torch.Tensor,
            planar_symmetries: Optional[torch.Tensor],
            axis_continue_symmetries: Optional[torch.Tensor],
            axis_discrete_symmetries: Optional[torch.Tensor]
    ) -> (int, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]):
        (idx, points, planar_symmetries,
         axis_continue_symmetries, axis_discrete_symmetries) = self.a_transform.inverse_transform(
            idx, points, planar_symmetries, axis_continue_symmetries, axis_discrete_symmetries
        )
        return idx, points, planar_symmetries, axis_continue_symmetries, axis_discrete_symmetries
