import math
import lightning
import torch
from lightning.pytorch.callbacks import EarlyStopping
from torch import nn, Tensor
from torch.utils.data import Subset, DataLoader

from src.metrics.mAP import get_mean_average_precision
from src.metrics.phc import get_matches_amount, get_phc
from src.model.decoders.plane_predictor_head import PredictionHead
from src.model.encoders.pointnet_encoder import PointNetEncoder
from src.model.losses.discrete_prediction_loss import calculate_loss
from src.model.losses.utils import calculate_cost_matrix_normals


class SimpleNet(nn.Module):
    def __init__(
            self,
            n_heads: int = 3,
            use_bn=False
    ):
        super().__init__()

        self.h = n_heads
        self.use_bn = use_bn
        self.encoder = PointNetEncoder(use_bn=self.use_bn)
        self.prediction_heads = nn.ModuleList(
            [PredictionHead(use_bn=self.use_bn) for _ in range(self.h)]
        )

    def forward(self, x):
        batch_size = x.shape[0]

        x = self.encoder(x)

        pred_list = []
        for head in self.prediction_heads:
            pred_list.append(head(x))

        predictions = torch.vstack(pred_list).view(batch_size, self.h, 7)

        predictions[:, :, -1] = torch.sigmoid(predictions[:, :, -1])
        predictions[:, :, 0:3] = torch.nn.functional.normalize(predictions[:, :, 0:3].clone(), dim=2)

        return predictions


class LightingMyNet(lightning.LightningModule):
    def __init__(self,
                 n_heads: int = 3,
                 weights=torch.tensor([1.0, 1.0, 1.0, 1.0]),
                 cost_matrix_method=calculate_cost_matrix_normals,
                 print_losses=False,
                 use_bn=False,
                 ):
        super().__init__()
        self.use_bn = use_bn
        self.print_losses = print_losses
        self.cost_matrix_method = cost_matrix_method
        self.losses_weights = weights

        self.net = SimpleNet(
            n_heads=n_heads,
            use_bn=use_bn
        )
        self.save_hyperparameters(ignore=["net"])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        idxs, points, sym_planes, transforms = batch
        points = torch.transpose(points, 1, 2).float()

        y_pred = self.net.forward(points)
        loss = calculate_loss(
            batch, y_pred,
            self.cost_matrix_method, self.losses_weights,
            self.print_losses
        )

        prediction = [(batch, y_pred)]
        mean_avg_precision = get_mean_average_precision(prediction)
        phc = get_phc(prediction)

        self.log("train_loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=len(sym_planes))
        self.log("train_MAP", mean_avg_precision, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=len(sym_planes))
        self.log("train_PHC", phc, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=len(sym_planes))
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        idxs, points, sym_planes, transforms = batch
        points = torch.transpose(points, 1, 2).float()

        y_pred = self.net.forward(points)
        loss = calculate_loss(
            batch, y_pred,
            self.cost_matrix_method, self.losses_weights,
            self.print_losses
        )

        self.log("val_loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=len(sym_planes))

        return loss

    def test_step(self, batch, batch_idx):
        idxs, points, sym_planes, transforms = batch
        points = torch.transpose(points, 1, 2).float()

        y_pred = self.net.forward(points)
        loss = calculate_loss(
            batch, y_pred,
            self.cost_matrix_method, self.losses_weights,
            self.print_losses
        )

        self.log("test_loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=len(sym_planes))

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        idxs, points, y_true, transforms = batch
        points = torch.transpose(points, 1, 2).float()

        y_pred = self.net.forward(points)

        return batch, y_pred

    def on_after_backward(self):
        for name, param in self.net.named_parameters():
            if param.grad.isnan().any():
                print(f"{name} got nan!")


if __name__ == "__main__":
    from src.dataset.shrec2023 import (SymmetryDataset,
                                       SymmetryDataModule,
                                       default_symmetry_dataset_collate_fn,
                                       default_symmetry_dataset_collate_fn_list_sym)
    from src.dataset.preprocessing import *

    DATA_PATH = "/data/shrec_2023/benchmark-train"
    BATCH_SIZE = 1
    EXAMPLES_USED = 10
    SAMPLE_SIZE = 1024
    COLLATE_FN = default_symmetry_dataset_collate_fn_list_sym

    scaler = UnitSphereNormalization()
    sampler = RandomSampler(sample_size=SAMPLE_SIZE, keep_copy=True)
    compose_transform = ComposeTransform([scaler])

    dataset = SymmetryDataset(DATA_PATH, compose_transform)
    datamodule = SymmetryDataModule(
        train_data_path=DATA_PATH,
        test_data_path=DATA_PATH,
        predict_data_path=DATA_PATH,
        does_predict_has_ground_truths=True,
        batch_size=BATCH_SIZE,
        transform=compose_transform,
        collate_function=COLLATE_FN,
        validation_percentage=0.9999,
        shuffle=True,
        n_workers=1,
    )

    test_net = LightingMyNet(27, use_bn=False, print_losses=False, weights=[1.0, 0.05, 1.0, 1.0])  #
    # mpath = "modelos_interesantes/simple_net/version_9/checkpoints/epoch_epoch=30_val_loss=0.47_train_loss=0.47.ckpt"
    # test_net = LightingMyNet.load_from_checkpoint(mpath)

    datamodule.setup("predict")
    datamodule.setup("fit")

    trainer = lightning.Trainer(
        fast_dev_run=False,
        limit_val_batches=0.0,
        enable_progress_bar=True,
        max_epochs=500,
        callbacks=[
            EarlyStopping("train_loss", patience=10, verbose=True)
        ]
    )
    predict_dataset = Subset(datamodule.predict_dataset, [i for i in range(EXAMPLES_USED)])
    predict_dataloader = DataLoader(predict_dataset, batch_size=BATCH_SIZE,
                                    collate_fn=COLLATE_FN)

    trainer.fit(test_net, predict_dataloader)
    predictions = trainer.predict(test_net, predict_dataloader)

    batch, y_pred = predictions[0]
    _, _, y_true, _ = batch

    print(y_true[0])
    print(y_pred[0])

    print("Metricas normales")
    print("PHC", get_phc(predictions).item())
    print("MAP", get_mean_average_precision(predictions).item())

    print("Match normales")
    print("PHC", get_phc(predictions, eps=1).item())
    print("MAP", get_mean_average_precision(predictions, eps=1).item())

    print("Match centros")
    print("PHC", get_phc(predictions, theta=10).item())
    print("MAP", get_mean_average_precision(predictions, theta=10).item())
