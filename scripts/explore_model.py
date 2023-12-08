import lightning
import torch

from src.dataset.preprocessing import ComposeTransform, RandomSampler, UnitSphereNormalization
from src.model.loss.center_loss import center_loss
from src.model.network import LightingSymmetryNet
from src.dataset.shrec2023 import SymmetryDataModule, default_symmetry_dataset_collate_fn, \
    default_symmetry_dataset_collate_fn_list_sym
from torch.utils.data import Dataset
from torch.utils.data import random_split, DataLoader, Subset
from lightning.pytorch.callbacks import EarlyStopping

if __name__ == "__main__":
    DATA_PATH = "/data/shrec_2023/benchmark-train"
    BATCH_SIZE = 1
    SAMPLE_SIZE = 1000
    COLLATE_FN = default_symmetry_dataset_collate_fn_list_sym

    scaler = UnitSphereNormalization()
    sampler = RandomSampler(sample_size=SAMPLE_SIZE, keep_copy=False)
    default_transform = ComposeTransform([scaler, sampler])

    datamodule = SymmetryDataModule(
        train_data_path=DATA_PATH,
        test_data_path=DATA_PATH,
        predict_data_path=DATA_PATH,
        does_predict_has_ground_truths=True,
        batch_size=BATCH_SIZE,
        collate_function=default_symmetry_dataset_collate_fn_list_sym,
        validation_percentage=0.9999,
        transform=default_transform,
        shuffle=True,
        n_workers=1,
    )
    datamodule.setup("predict")

    lnet = LightingSymmetryNet(
        batch_size=BATCH_SIZE,
        num_points=SAMPLE_SIZE,
        n_prediction_per_point=1,
        n_heads=1,
        loss_fn=center_loss
    )
    trainer = lightning.Trainer(
        fast_dev_run=False,
        limit_val_batches=0.0,
        enable_progress_bar=True,
        max_epochs=500,
        callbacks=[
            EarlyStopping("train_loss", patience=25, verbose=True)
        ]
    )
    predict_dataset = Subset(datamodule.predict_dataset, [i for i in range(BATCH_SIZE)])
    predict_dataloader = DataLoader(predict_dataset, batch_size=BATCH_SIZE,
                                    collate_fn=COLLATE_FN)
    trainer.fit(lnet, predict_dataloader)
    preds, y_true = trainer.predict(lnet, predict_dataloader)[0]
    print(y_true[0][:, 3:6])
    print("==")
    print(preds[0][0][0])

    print()

