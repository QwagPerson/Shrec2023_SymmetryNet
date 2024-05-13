import lightning
import torch
from torch import nn

from src.dataset.SymDataModule import custom_collate_fn, SymDataModule
from src.dataset.transforms.ComposeTransform import ComposeTransform
from src.dataset.transforms.RandomSampler import RandomSampler
from src.dataset.transforms.UnitSphereNormalization import UnitSphereNormalization
from src.metrics.MAP import get_mean_average_precision
from src.metrics.PHC import get_phc
from src.model.CenterNNormalsNet import CenterNNormalsNet
from src.model.LightingCenterNNormalsNet import LightingCenterNNormalsNet
from src.model.encoders.PCT import PCT
from src.model.encoders.PointNetPlusPlusEncoder import PointNetPlusPlusEncoder
from src.model.encoders.pointnet_encoder import PointNetEncoder


class ClassificationModel(nn.Module):
    def __init__(self, encoder: str = "pointnet", n_classes: int = 10):
        super().__init__()
        self.n_classes = n_classes

        if encoder == "pointnet":
            self.encoder = PointNetEncoder()
            self.encoder_output_size = 1024
        elif encoder == "pointnetplusplus":
            self.encoder = PointNetPlusPlusEncoder()
            self.encoder_output_size = 1024
        elif encoder == "PCT":
            self.encoder = PCT()
            self.encoder_output_size = 1024
        else:
            raise ValueError("Encoder not found")

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.n_classes)

        self.act_fun = nn.ReLU()
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.encoder(x)
        x = self.dropout(self.act_fun(self.fc1(x)))
        x = self.dropout(self.act_fun(self.fc2(x)))
        x = self.fc3(x)
        return x


class ClassificationTask(lightning.LightningModule):
    def __init__(self, encoder: str = "pointnet"):
        super().__init__()
        self.encoder_used = encoder

        self.net = ClassificationModel(encoder=encoder)
        self.loss_fun = nn.CrossEntropyLoss()
        self.save_hyperparameters(ignore=["net"])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

    def _log(
            self, metric_val, metric_name, step_tag, batch_size,
            on_step=True, on_epoch=True, prog_bar=False, sync_dist=True
    ):
        self.log(f"{step_tag}_{metric_name}", metric_val, on_step=on_step, on_epoch=on_epoch,
                 prog_bar=prog_bar, batch_size=batch_size, sync_dist=sync_dist)

    def _step(self, batch, step_tag):
        batch.device = self.device

        points = torch.stack(batch.get_points())
        points = torch.transpose(points, 1, 2).float()

        logits = self.net(points)
        y_true = batch.get_shape_type_classification_labels()

        loss = self.loss_fun(logits, y_true)


        predicted_label = logits.max(dim=1).indices
        true_label = y_true.max(dim=1).indices

        acc = (predicted_label == true_label).sum() / batch.size

        self._log(loss, "loss", step_tag, batch.size, prog_bar=False)
        self._log(acc, "acc", step_tag, batch.size, prog_bar=True, on_step=False)

        return loss

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        batch.device = self.device
        self.matcher.device = self.device

        points = torch.stack(batch.get_points())
        points = torch.transpose(points, 1, 2).float()

        plane_predictions, axis_discrete_predictions, axis_continue_predictions = self.net.forward(points)

        return batch, plane_predictions, axis_discrete_predictions, axis_continue_predictions

    def on_after_backward(self):
        for name, param in self.net.named_parameters():
            if param.grad is not None:
                if param.grad.isnan().any():
                    print(f"{name} got nan!")


if __name__ == "__main__":
    DATA_PATH = "/data/temp"  # "/data/sym-10k-xz-split-class-noparallel/"
    BATCH_SIZE = 2
    PREDICT_SAMPLES = 1
    SAMPLE_SIZE = 14_440
    COLLATE_FN = custom_collate_fn
    NUM_WORKERS = 15

    scaler = UnitSphereNormalization()
    sampler = RandomSampler(sample_size=SAMPLE_SIZE, keep_copy=True)
    # compose_transform = ComposeTransform([sampler, scaler])
    compose_transform = ComposeTransform([scaler])

    datamodule = SymDataModule(
        dataset_path=DATA_PATH,
        predict_data_path=DATA_PATH,
        does_predict_has_ground_truths=True,
        batch_size=BATCH_SIZE,
        transform=compose_transform,
        collate_function=custom_collate_fn,
        shuffle=True,
        n_workers=1,
    )
    datamodule.setup("predict")
    datamodule.setup("fit")
    print(len(datamodule.train_dataloader()))

    test_net = ClassificationTask(encoder="pointnet")

    test_batch = next(iter(datamodule.train_dataloader()))

    test_net.training_step(batch=test_batch, batch_idx=0, dataloader_idx=0)

    trainer = lightning.Trainer(enable_progress_bar=True, logger=False)
    trainer.fit(test_net, datamodule)