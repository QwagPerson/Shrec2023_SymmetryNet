import torch
from torch import nn

from src.model.decoders.center_prediction_head import CenterPredictionHead
from src.model.decoders.prediction_head import PredictionHead
from src.model.encoders.pointnet_encoder import PointNetEncoder


from src.model.encoders.pointnext_encoder_parameters import *
from src.model.encoders.pointnext_encoder import PointNeXt



class CenterNNormalsNet(nn.Module):
    def __init__(
            self,
            amount_of_plane_normals_predicted=10,
            amount_of_axis_discrete_normals_predicted=10,
            amount_of_axis_continue_normals_predicted=10,
            use_bn=False,
            normalize_normals=False,
            print_losses=False,
            pointnext_encoder='None',		# 'PointNeXt_B' (21.5 M), 'PointNeXt_L2' (32.0 M), 'PointNeXt_XXL' (73.8 M)
    ):
        super().__init__()
        self.use_bn = use_bn
        self.normalize_normals = normalize_normals
        self.amount_plane_normals = amount_of_plane_normals_predicted
        self.amount_axis_discrete_normals = amount_of_axis_discrete_normals_predicted
        self.amount_axis_continue_normals = amount_of_axis_continue_normals_predicted
        self.print_losses = print_losses
        #self.use_pointnext = use_pointnext
        self.pointnext_encoder = pointnext_encoder

        if self.pointnext_encoder != 'None':
            model_cfg = MODEL_CONFIG[self.pointnext_encoder]
            self.encoder = PointNeXt(model_cfg) # .to(device=args.device)
            print(f'PointNeXt encoder {pointnext_encoder}: {self.encoder}')
        else:
            self.encoder = PointNetEncoder(use_bn=self.use_bn)
            print(f'PointNet encoder: {self.encoder}')

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

    def _forward_plane_normals(self, batch_size, pcd_features, center):
        plane_normals_list = []

        for head in self.plane_normals_heads:
            plane_normals_list.append(head(pcd_features))

        plane_normals = (torch.vstack(plane_normals_list).view(
            batch_size, self.amount_plane_normals, 4
        ))
        plane_predictions = torch.concat(
            (plane_normals, center.repeat(1, self.amount_plane_normals, 1)), dim=2
        )

        reorder_planes = torch.tensor([0, 1, 2, 4, 5, 6, 3], device=plane_predictions.device).long()
        plane_predictions = plane_predictions[:, :, reorder_planes]
        plane_predictions[:, :, -1] = torch.sigmoid(plane_predictions[:, :, -1])

        if self.normalize_normals:
            plane_predictions[:, :, 0:3] = torch.nn.functional.normalize(
                plane_predictions[:, :, 0:3].clone(), dim=2
            )

        return plane_predictions

    def _forward_axis_discrete_normals(self, batch_size, pcd_features, center):
        axis_discrete_normals_list = []

        for head in self.axis_discrete_normals_heads:
            axis_discrete_normals_list.append(head(pcd_features))

        axis_discrete_normals = (torch.vstack(axis_discrete_normals_list).view(
            batch_size, self.amount_axis_discrete_normals, 5
        ))
        axis_discrete_predictions = torch.concat(
            (axis_discrete_normals, center.repeat(1, self.amount_axis_discrete_normals, 1)), dim=2
        )
        reorder_axis_discrete = torch.tensor([0, 1, 2, 4, 5, 6, 7, 3], device=axis_discrete_predictions.device).long()
        axis_discrete_predictions = axis_discrete_predictions[:, :, reorder_axis_discrete]
        axis_discrete_predictions[:, :, -1] = torch.sigmoid(axis_discrete_predictions[:, :, -1])

        if self.normalize_normals:
            axis_discrete_predictions[:, :, 0:3] = torch.nn.functional.normalize(
                axis_discrete_predictions[:, :, 0:3].clone(), dim=2
            )

        return axis_discrete_predictions

    def _forward_axis_continue(self, batch_size, pcd_features, center):
        axis_continue_normals_list = []

        for head in self.axis_continue_normals_heads:
            axis_continue_normals_list.append(head(pcd_features))

        axis_continue_normals = (torch.vstack(axis_continue_normals_list).view(
            batch_size, self.amount_axis_continue_normals, 4
        ))
        axis_continue_predictions = torch.concat(
            (axis_continue_normals, center.repeat(1, self.amount_axis_continue_normals, 1)), dim=2
        )

        reorder_planes = torch.tensor([0, 1, 2, 4, 5, 6, 3], device=axis_continue_predictions.device).long()
        axis_continue_predictions = axis_continue_predictions[:, :, reorder_planes]
        axis_continue_predictions[:, :, -1] = torch.sigmoid(axis_continue_predictions[:, :, -1])

        if self.normalize_normals:
            axis_continue_predictions[:, :, 0:3] = torch.nn.functional.normalize(
                axis_continue_predictions[:, :, 0:3].clone(), dim=2
            )

        return axis_continue_predictions

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.encoder(x)
        '''
        print(f'x.shape: {x.shape}')
        if self.use_pointnext and False:
            x = x.transpose(1, 2)
        print(f'x.shape: {x.shape}')
        '''

        center = self.center_prediction_head(x).unsqueeze(dim=1)

        plane_predictions = (self._forward_plane_normals(batch_size, x, center)
                             if self.amount_plane_normals > 0 else None)

        axis_discrete_predictions = (self._forward_axis_discrete_normals(batch_size, x, center)
                                     if self.amount_axis_discrete_normals > 0 else None)

        axis_continue_predictions = (self._forward_axis_continue(batch_size, x, center)
                                     if self.amount_axis_continue_normals > 0 else None)

        return plane_predictions, axis_discrete_predictions, axis_continue_predictions
