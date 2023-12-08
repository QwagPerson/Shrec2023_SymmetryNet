import math
import lightning
import torch
from lightning.pytorch.callbacks import EarlyStopping
from torch import nn
from torch.utils.data import Subset, DataLoader

from src.metrics.phc import calculate_phc
from src.model.my_net.decoder.prediction_head import PredictionHead
from src.model.my_net.encoder.pointnet_encoder import PointNetEncoder
from src.model.my_net.loss.plane_nearness import calculate_loss
from src.model.my_net.loss.symmetry_distance_error import calculate_loss_sde


class MyNet(nn.Module):
    def __init__(
            self,
            n_heads: int = 3,
    ):
        super().__init__()

        self.h = n_heads

        self.encoder = PointNetEncoder()
        self.prediction_heads = nn.ModuleList(
            [PredictionHead() for _ in range(self.h)]
        )

    def forward(self, x):
        batch_size = x.shape[0]
        if x.isnan().any():
            print("AAAAAAAAAAAAAAA")
        x1 = self.encoder(x)
        if x1.isnan().any():
            print("BBB")
        pred_list = []
        for head in self.prediction_heads:
            pred_list.append(head(x1))
        predictions = torch.vstack(pred_list).view(batch_size, self.h, 7)
        if predictions.isnan().any():
            print("CCC")
        predictions[:, :, -1] = torch.sigmoid(predictions[:, :, -1])
        predictions[:, :, 0:3] = torch.nn.functional.normalize(predictions[:, :, 0:3].clone(), dim=2)

        return predictions


class LightingMyNet(lightning.LightningModule):
    def __init__(self,
                 n_heads: int = 3,
                 loss_fn=None,
                 ):
        super().__init__()
        self.net = MyNet(
            n_heads=n_heads,
        )
        self.loss_fn = loss_fn
        self.save_hyperparameters(ignore=["net", "loss_fn"])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        idxs, points, sym_planes, transforms = batch
        points = torch.transpose(points, 1, 2).float()
        y_pred = self.net.forward(points)
        loss = self.loss_fn(batch, y_pred)

        self.log("train_loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=len(sym_planes))
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        idxs, points, sym_planes, transforms = batch
        points = torch.transpose(points, 1, 2).float()

        y_pred = self.net.forward(points)
        loss = self.loss_fn(batch, y_pred)

        self.log("val_loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=len(sym_planes))

        return loss

    def test_step(self, batch, batch_idx):
        idxs, points, sym_planes, transforms = batch
        points = torch.transpose(points, 1, 2).float()

        y_pred = self.net.forward(points)
        loss = self.loss_fn(batch, y_pred)

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
                print(f"{name}: {param.grad}")



if __name__ == "__main__":
    from src.dataset.shrec2023 import (SymmetryDataset,
                                       SymmetryDataModule,
                                       default_symmetry_dataset_collate_fn,
                                       default_symmetry_dataset_collate_fn_list_sym)
    from src.dataset.preprocessing import *

    DATA_PATH = "/data/shrec_2023/benchmark-train"
    BATCH_SIZE = 10
    SAMPLE_SIZE = 4096
    COLLATE_FN = default_symmetry_dataset_collate_fn_list_sym
    scaler = UnitSphereNormalization()
    sampler = RandomSampler(sample_size=SAMPLE_SIZE, keep_copy=True)
    compose_transform = ComposeTransform([scaler, sampler])

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

    test_net = LightingMyNet(20, loss_fn=calculate_loss) #
    #test_net = LightingMyNet.load_from_checkpoint("lightning_logs/version_33/checkpoints/epoch=49-step=50.ckpt")

    datamodule.setup("predict")

    torch.autograd.set_detect_anomaly(True)
    trainer = lightning.Trainer(
        fast_dev_run=False,
        limit_val_batches=0.0,
        enable_progress_bar=True,
        max_epochs=500,
        callbacks=[
            EarlyStopping("train_loss", patience=10, verbose=True)
        ]
    )
    predict_dataset = Subset(datamodule.predict_dataset, [i for i in range(BATCH_SIZE)])
    predict_dataloader = DataLoader(predict_dataset, batch_size=BATCH_SIZE,
                                    collate_fn=COLLATE_FN)

    trainer.fit(test_net, predict_dataloader)
    predictions = trainer.predict(test_net, predict_dataloader)

    good_matches = 0

    for (batch, y_pred ) in predictions:
        good_matches += calculate_phc(
            batch,
            y_pred,
        )

    print(good_matches/(BATCH_SIZE*len(predictions)))
