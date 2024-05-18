from typing import Optional
import torch

from src.dataset.transforms.AbstractTransform import AbstractTransform


class UnitSphereNormalization(AbstractTransform):
    def __init__(self):
        self.centroid = None
        self.farthest_distance = None

    def _validate_self_attributes_are_not_none(self) -> Optional[Exception]:
        if self.centroid is None or self.farthest_distance is None:
            raise Exception(f"Transform variables where null when trying to execute a method that needs them."
                            f"Variables; Centroid: {self.centroid} | Farthest distance {self.farthest_distance}")
        return None

    def _normalize_points(self, points: torch.Tensor) -> torch.Tensor:
        self.centroid = torch.mean(points, dim=0)
        points = points - self.centroid
        self.farthest_distance = torch.max(torch.linalg.norm(points, dim=1))
        points = points / self.farthest_distance
        return points

    def _normalize_planes(self, symmetries: torch.Tensor) -> torch.Tensor:
        self._validate_self_attributes_are_not_none()
        symmetries[:, 3:6] = (symmetries[:, 3:6] - self.centroid) / self.farthest_distance
        return symmetries

    def _inverse_normalize_points(self, points: torch.Tensor) -> torch.Tensor:
        self._validate_self_attributes_are_not_none()
        points = (points * self.farthest_distance) + self.centroid
        return points

    def _inverse_normalize_planes(self, symmetries: torch.Tensor) -> torch.Tensor:
        self._validate_self_attributes_are_not_none()
        symmetries[:, 3:6] = (symmetries[:, 3:6] * self.farthest_distance) + self.centroid
        return symmetries

    def _handle_device(self, device):
        self.centroid = self.centroid.to(device)
        self.farthest_distance = self.farthest_distance.to(device)

    def inverse_transform(
            self,
            idx: int,
            points: torch.Tensor,
            planar_symmetries: Optional[torch.Tensor],
            axis_continue_symmetries: Optional[torch.Tensor],
            axis_discrete_symmetries: Optional[torch.Tensor]
    ) -> (int, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]):
        self._validate_self_attributes_are_not_none()
        self._handle_device(points.device)
        points = self._inverse_normalize_points(points)

        if planar_symmetries is not None:
            planar_symmetries = self._inverse_normalize_planes(planar_symmetries)
        if axis_continue_symmetries is not None:
            axis_continue_symmetries = self._inverse_normalize_planes(axis_continue_symmetries)
        if axis_discrete_symmetries is not None:
            axis_discrete_symmetries = self._inverse_normalize_planes(axis_discrete_symmetries)

        return idx, points, planar_symmetries, axis_continue_symmetries, axis_discrete_symmetries

    def transform(
            self,
            idx: int,
            points: torch.Tensor,
            planar_symmetries: Optional[torch.Tensor],
            axis_continue_symmetries: Optional[torch.Tensor],
            axis_discrete_symmetries: Optional[torch.Tensor]
    ) -> (int, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]):
        points = self._normalize_points(points)
        if planar_symmetries is not None:
            planar_symmetries = self._normalize_planes(planar_symmetries)
        if axis_continue_symmetries is not None:
            axis_continue_symmetries = self._normalize_planes(axis_continue_symmetries)
        if axis_discrete_symmetries is not None:
            axis_discrete_symmetries = self._normalize_planes(axis_discrete_symmetries)
        return idx, points, planar_symmetries, axis_continue_symmetries, axis_discrete_symmetries
