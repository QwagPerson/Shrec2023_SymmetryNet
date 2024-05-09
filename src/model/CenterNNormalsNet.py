import torch
from torch import nn

from src.model.decoders.center_prediction_head import CenterPredictionHead
from src.model.decoders.prediction_head import PredictionHead
from src.model.encoders.pointnet_encoder import PointNetEncoder


class CenterNNormalsNet(nn.Module):
    def __init__(
            self,
            amount_of_plane_normals_predicted=10,
            amount_of_axis_discrete_normals_predicted=10,
            amount_of_axis_continue_normals_predicted=10,
            use_bn=False,
            normalize_normals=False,
    ):
        super().__init__()
        self.use_bn = use_bn
        self.normalize_normals = normalize_normals
        self.amount_plane_normals = amount_of_plane_normals_predicted
        self.amount_axis_discrete_normals = amount_of_axis_discrete_normals_predicted
        self.amount_axis_continue_normals = amount_of_axis_continue_normals_predicted

        self.encoder = PointNetEncoder(use_bn=self.use_bn)

        # nx ny nz & confidence
        self.plane_normals_heads = nn.ModuleList(
            [PredictionHead(output_size=4, use_bn=self.use_bn) for _ in range(self.amount_plane_normals)]
        )

        # nx ny nz theta & confidence
        self.axis_discrete_normals_heads = nn.ModuleList(
            [PredictionHead(output_size=5, use_bn=self.use_bn) for _ in range(self.amount_axis_discrete_normals)]
        )

        # nx ny nz & confidence
        self.axis_continue_normals_heads = nn.ModuleList(
            [PredictionHead(output_size=4, use_bn=self.use_bn) for _ in range(self.amount_axis_continue_normals)]
        )

        self.center_prediction_head = CenterPredictionHead(use_bn=self.use_bn)

    def forward(self, x):
        batch_size = x.shape[0]
        plane_normals_list = []
        axis_discrete_normals_list = []
        axis_continue_normals_list = []

        x = self.encoder(x)
        center = self.center_prediction_head(x).unsqueeze(dim=1)

        for head in self.plane_normals_heads:
            plane_normals_list.append(head(x))

        for head in self.axis_discrete_normals_heads:
            axis_discrete_normals_list.append(head(x))

        for head in self.axis_continue_normals_heads:
            axis_continue_normals_list.append(head(x))

        # Plane prediction
        # Normal (3) + Confidence(1)
        if self.amount_plane_normals > 0:
            plane_normals = (torch.vstack(plane_normals_list).view(
                batch_size, self.amount_plane_normals, 4
            ))
            plane_predictions = torch.concat(
                (plane_normals, center.repeat(1, self.amount_plane_normals, 1)), dim=2
            )

            reorder_planes = torch.tensor([0, 1, 2, 4, 5, 6, 3], device=plane_predictions.device).long()
            plane_predictions = plane_predictions[:, :, reorder_planes]
            plane_predictions[:, :, -1] = torch.sigmoid(plane_predictions[:, :, -1])
        else:
            plane_predictions = None

        # Axis discrete prediction
        # Normal (3) + Theta (1) + Confidence(1)
        if self.amount_axis_discrete_normals > 0:
            axis_discrete_normals = (torch.vstack(axis_discrete_normals_list).view(
                batch_size, self.amount_axis_discrete_normals, 5
            ))
            axis_discrete_predictions = torch.concat(
                (axis_discrete_normals, center.repeat(1, self.amount_axis_discrete_normals, 1)), dim=2
            )
            reorder_axis_discrete = torch.tensor([0, 1, 2, 4, 5, 6, 7, 3], device=plane_predictions.device).long()
            axis_discrete_predictions = axis_discrete_predictions[:, :, reorder_axis_discrete]
            axis_discrete_predictions[:, :, -1] = torch.sigmoid(axis_discrete_predictions[:, :, -1])
        else:
            axis_discrete_predictions = None

        # Axis continue prediction
        # Normal (3) + Confidence(1)
        if self.amount_axis_continue_normals > 0:
            axis_continue_normals = (torch.vstack(axis_continue_normals_list).view(
                batch_size, self.amount_axis_continue_normals, 4
            ))
            axis_continue_predictions = torch.concat(
                (axis_continue_normals, center.repeat(1, self.amount_axis_continue_normals, 1)), dim=2
            )

            reorder_planes = torch.tensor([0, 1, 2, 4, 5, 6, 3], device=axis_continue_predictions.device).long()
            axis_continue_predictions = axis_continue_predictions[:, :, reorder_planes]
            axis_continue_predictions[:, :, -1] = torch.sigmoid(axis_continue_predictions[:, :, -1])
        else:
            axis_continue_predictions = None

        if self.normalize_normals:
            if plane_predictions is not None:
                plane_predictions[:, :, 0:3] = torch.nn.functional.normalize(
                    plane_predictions[:, :, 0:3].clone(), dim=2
                )
            if axis_discrete_predictions is not None:
                axis_discrete_predictions[:, :, 0:3] = torch.nn.functional.normalize(
                    axis_discrete_predictions[:, :, 0:3].clone(), dim=2
                )
            if axis_continue_predictions is not None:
                axis_continue_predictions[:, :, 0:3] = torch.nn.functional.normalize(
                    axis_continue_predictions[:, :, 0:3].clone(), dim=2
                )

        return plane_predictions, axis_discrete_predictions, axis_continue_predictions
