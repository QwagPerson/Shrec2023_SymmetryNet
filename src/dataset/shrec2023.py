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

if __name__ == "__main__":
    import sys

    sys.path.insert(0, '../..')

from src.dataset.preprocessing import Shrec2023Transform, UnitSphereNormalization, RandomSampler


def parse_sym_file(fname):
    planar_symmetries = []
    axis_continue_symmetries = []
    axis_discrete_symmetries = []

    with open(fname) as f:
        line_amount = int(f.readline())
        for _ in range(line_amount):
            line = f.readline().split(" ")
            line = [x.replace("\n", "") for x in line]
            if line[0] == "plane":
                plane = [float(x)for x in line[1::]]
                planar_symmetries.append(torch.tensor(plane))
            elif line[0] == "axis" and line[-1] == "inf":
                plane = [float(x) for x in line[1:7]]
                axis_continue_symmetries.append(torch.tensor(plane))
            else:
                plane = [float(x) for x in line[1::]]
                axis_discrete_symmetries.append(torch.tensor(plane))

    planar_symmetries = None if len(planar_symmetries) == 0 else torch.stack(planar_symmetries).float()
    axis_continue_symmetries = None if len(axis_continue_symmetries) == 0 else torch.stack(axis_continue_symmetries).float()
    axis_discrete_symmetries = None if len(axis_discrete_symmetries) == 0 else torch.stack(axis_discrete_symmetries).float()

    return planar_symmetries, axis_continue_symmetries, axis_discrete_symmetries


def default_symmetry_dataset_collate_fn_list_sym(batch):
    idxs = torch.tensor([item[0] for item in batch])
    points = torch.stack([item[1] for item in batch])
    planar_syms = [item[2] for item in batch]
    axis_continue_syms = [item[3] for item in batch]
    axis_discrete_syms = [item[4] for item in batch]
    transforms = [item[5] for item in batch]
    return idxs, points, planar_syms, axis_continue_syms, axis_discrete_syms, transforms


def unsqueeze_if_only_one_symmetry(symmetries):
    if len(symmetries.shape) == 1:
        return symmetries.unsqueeze(0)
    else:
        return symmetries


class SymmetryDataset(Dataset):
    def __init__(
            self,
            data_source_path: str = "path/to/dataset/split",
            transform: Optional[Shrec2023Transform] = None,
            has_ground_truth: bool = True,
            debug=False
    ):
        """
        Dataset used for a track of SHREC2023. It contains a set of 3D points
        and planes that represent reflective symmetries.
        :param data_source_path: Path to folder that contains the points and symmetries.
        :param transform: Transform applied to dataset item.
        """
        self.data_source_path = Path(data_source_path)
        self.transform = transform
        self.has_ground_truth = has_ground_truth
        self.debug = debug

        if self.debug:
            print(f'Searching xz-compressed point cloud files in {self.data_source_path}...')
        self.flist = list(self.data_source_path.rglob(f'*/*.xz'))
        self.length = len(self.flist)
        if self.debug:
            print(f'{self.data_source_path.name}: found {self.length} files:\n{self.flist[:5]}\n{self.flist[-5:]}\n')

    def fname_from_idx(self, idx: int) -> Tuple[Path, str]:
        if idx < 0 or idx >= len(self.flist):
            raise IndexError(f"Invalid index: {idx}, dataset size is: {len(self.flist)}")
        fname = self.flist[idx]
        if self.debug:
            print(f'Opening file: {fname.name}')
        return fname, str(fname).replace('.xz', '-sym.txt')

    def read_points(self, idx: int) -> torch.Tensor:
        """
        Reads the points with index idx.
        :param idx: Index of points to be read. Not to be confused with the shape ID, this is now just the index in self.flist
        :return: A tensor of shape N x 3 where N is the amount of points.
        """
        fname, _ = self.fname_from_idx(idx)

        with lzma.open(fname, 'rb') as fhandle:
            points = torch.tensor(np.loadtxt(fhandle))

        if self.debug:
            torch.set_printoptions(linewidth=200)
            torch.set_printoptions(precision=3)
            torch.set_printoptions(sci_mode=False)
            print(f'[{idx}]: {points.shape = }\n{points = }')

        return points

    def read_planes(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Read symmetry planes from file with its first line being the number of symmetry planes
        and the rest being the symmetry planes.
        :param idx: The idx of the syms to reads
        :return: A tensor of planes represented by their normals and points. N x 6 where
        N is the amount of planes and 6 because the first 3 elements
        are the normal and the last 3 are the point.
        """
        _, sym_fname = self.fname_from_idx(idx)
        return parse_sym_file(sym_fname)

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> (int, torch.Tensor, Optional[torch.Tensor], List[Shrec2023Transform]):
        points = self.read_points(idx)
        planar_symmetries = None
        axis_continue_symmetries = None
        axis_discrete_symmetries = None

        if self.has_ground_truth:
            planar_symmetries, axis_continue_symmetries, axis_discrete_symmetries = self.read_planes(idx)

        if self.transform is not None:
            idx, points, planar_symmetries, axis_continue_symmetries, axis_discrete_symmetries = self.transform(
                idx, points, planar_symmetries, axis_continue_symmetries, axis_discrete_symmetries
            )

        transform_used = copy.deepcopy(self.transform)
        return (idx, points.float(),
                planar_symmetries, axis_continue_symmetries, axis_discrete_symmetries,
                transform_used)


scaler = UnitSphereNormalization()
sampler = RandomSampler(sample_size=1024, keep_copy=False)
# default_transform = ComposeTransform([scaler, sampler])
default_transform = scaler


class SymmetryDataModule(lightning.LightningDataModule):
    def __init__(
            self,
            dataset_path: str = "/path/to/dataset",
            predict_data_path: str = "/path/to/predict_data",
            does_predict_has_ground_truths: bool = False,
            batch_size: int = 2,
            transform: Optional[Shrec2023Transform] = None,
            collate_function=None,
            shuffle: bool = True,
            n_workers: int = 1,
    ):
        """
        Data module designed to load Shrec2023 symmetry dataset.
        :param dataset_path: Path to dataset, it must contain points and planes split in train/valid/test sets.
        :param predict_data_path: Path to predict it must contain points and can contain planes
        :param does_predict_has_ground_truths : Boolean flag to indicate if the predict data has
        ground truths.
        :param batch_size: Batch size used on all dataloaders.
        :param transform: Transform applied to all dataloaders.
        :param collate_function: Function used to batch the items from symmetry dataset.
        :param shuffle: True if you want to shuffle the train dataloader every epoch.
        :param n_workers: Amount of workers used for loading data into RAM.
        """
        super().__init__()
        self.dataset_path = dataset_path
        self.predict_data_path = predict_data_path
        self.does_predict_has_ground_truths = does_predict_has_ground_truths
        self.batch_size = batch_size
        self.transform = default_transform if transform is None else transform
        if collate_function is None:
            self.collate_function = default_symmetry_dataset_collate_fn_list_sym
        else:
            self.collate_function = collate_function
        self.shuffle = shuffle
        self.n_workers = n_workers

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = SymmetryDataset(
                data_source_path=Path(self.dataset_path) / 'train',
                transform=self.transform,
                has_ground_truth=True
            )
            self.valid_dataset = SymmetryDataset(
                data_source_path=Path(self.dataset_path) / 'valid',
                transform=self.transform,
                has_ground_truth=True
            )

        if stage == "test":
            self.test_dataset = SymmetryDataset(
                data_source_path=Path(self.dataset_path) / 'test',
                transform=self.transform,
                has_ground_truth=True
            )

        if stage == "predict":
            self.predict_dataset = SymmetryDataset(
                data_source_path=self.predict_data_path,
                transform=self.transform,
                has_ground_truth=self.does_predict_has_ground_truths
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            collate_fn=self.collate_function,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.n_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            collate_fn=self.collate_function,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            collate_fn=self.collate_function,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            collate_fn=self.collate_function,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_workers,
        )


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
