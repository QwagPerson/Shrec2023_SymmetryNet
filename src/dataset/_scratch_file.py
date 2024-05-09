import copy
import lzma
from pathlib import Path
from typing import Optional, List, Tuple

import lightning
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from src.dataset.transforms.AbstractTransform import AbstractTransform




if __name__ == "__main__":
    from src.dataset.preprocessing import *

    DATA_PATH = "/data/sym-10k-xz-split-class-noparallel"
    # DATA_PATH = "/mnt/btrfs-big/dataset/geometric-primitives-classification/symmetry-datasets/sym-10k-xz-split-class-noparallel"

    scaler = UnitSphereNormalization()
    sampler = RandomSampler(sample_size=3, keep_copy=True)
    default_transform = ComposeTransform([scaler, sampler])

    train_dataset = SymmetryDataset(Path(DATA_PATH) / 'train', default_transform)
    valid_dataset = SymmetryDataset(Path(DATA_PATH) / 'valid', default_transform)
    test_dataset = SymmetryDataset(Path(DATA_PATH) / 'test', default_transform)

    example_idx, example_points, example_syms_0, example_syms_1, example_syms_2, example_tr = train_dataset[724]

    if example_syms_0.shape[0] != 0:
        example_syms = example_syms_0
    elif example_syms_1.shape[0] != 0:
        example_syms = example_syms_1
    else:
        example_syms = example_syms_2

    print("transformed", example_idx, example_points[0, :], example_syms[0, :], example_tr)
    ridx, rpoints, sym1, sym2, sym3 = example_tr.inverse_transform(example_idx, example_points, example_syms_0,
                                                                   example_syms_1, example_syms_2)
    print("inverse_transform", ridx, rpoints[0, :])
    ridx, rpoints, sym1, sym2, sym3 = example_tr.transform(ridx, rpoints, sym1, sym2, sym3)
    print("transformed 2", ridx, rpoints[0, :])

    datamodule = SymmetryDataModule(
        dataset_path=DATA_PATH,
        predict_data_path=DATA_PATH,
        does_predict_has_ground_truths=True,
        batch_size=25,
        transform=default_transform,
        collate_function=default_symmetry_dataset_collate_fn_list_sym,
        shuffle=True,
        n_workers=1,
    )
    datamodule.setup("fit")

    train_dataloader = datamodule.train_dataloader()
    print(f'The training dataloader has: {len(train_dataloader)} batches')

    batch = next(iter(train_dataloader))

    # quick visualize to check the data.
    from scripts.visualize_prediction import visualize_prediction

    visualize_prediction(example_syms, example_points, example_syms)

    for z in batch:
        if isinstance(z, torch.Tensor):
            print(z.shape)
        else:
            print(len(z))
