import lightning
import torch

from src.dataset.preprocessing import ComposeTransform, RandomSampler, UnitSphereNormalization
from src.model.network import LightingSymmetryNet
from src.dataset.shrec2023 import SymmetryDataModule, default_symmetry_dataset_collate_fn
from torch.utils.data import Dataset
from torch.utils.data import random_split, DataLoader, Subset
if __name__ == "__main__":
    DATA_PATH = "/data/shrec_2023/benchmark-train"
    BATCH_SIZE = 1
    SAMPLE_SIZE = 10

    scaler = UnitSphereNormalization()
    sampler = RandomSampler(sample_size=SAMPLE_SIZE, keep_copy=True)
    compose_transform = ComposeTransform([scaler, sampler])


    datamodule = SymmetryDataModule(
        train_data_path=DATA_PATH,
        test_data_path=DATA_PATH,
        predict_data_path=DATA_PATH,
        does_predict_has_ground_truths=True,
        batch_size=BATCH_SIZE,
        transform=compose_transform,
        collate_function=default_symmetry_dataset_collate_fn,
        validation_percentage=0.9999,
        shuffle=True,
        n_workers=1,
    )
    datamodule.setup("predict")


    lnet = LightingSymmetryNet(BATCH_SIZE, SAMPLE_SIZE, 10)
    trainer = lightning.Trainer(fast_dev_run=False, limit_val_batches=0.0, enable_progress_bar=True)
    predict_dataset = Subset(datamodule.predict_dataset, [0])
    predict_dataloader = DataLoader(predict_dataset, batch_size=BATCH_SIZE, collate_fn=default_symmetry_dataset_collate_fn)
    out = trainer.predict(lnet, predict_dataloader)
    print(out)