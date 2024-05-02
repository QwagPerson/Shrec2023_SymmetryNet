import copy
import os
from typing import Optional, Callable, Tuple, List

import lzma
from pathlib import Path
import pandas as pd

import lightning
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split, DataLoader

if __name__ == "__main__":
        import sys 
        sys.path.insert(0, '../..')

from src.dataset.preprocessing import Shrec2023Transform, UnitSphereNormalization, RandomSampler, ComposeTransform


def default_symmetry_dataset_collate_fn(batch):
    idxs = torch.tensor([item[0] for item in batch])
    points = torch.stack([item[1] for item in batch])
    sym_planes = torch.nn.utils.rnn.pad_sequence([item[2] for item in batch], batch_first=True)
    transforms = [item[3] for item in batch]
    return idxs, points, sym_planes, transforms


def default_symmetry_dataset_collate_fn_list_sym(batch):
    idxs = torch.tensor([item[0] for item in batch])
    points = torch.stack([item[1] for item in batch])
    sym_planes = [item[2] for item in batch]
    transforms = [item[3] for item in batch]
    return idxs, points, sym_planes, transforms


class SymmetryDataset(Dataset):
    def __init__(
            self,
            data_source_path: str = "path/to/dataset/split",
            transform: Optional[Shrec2023Transform] = None,
            has_ground_truth: bool = True,
            debug = False
    ):
        """
        Dataset used for a track of SHREC2023. It contains a set of 3D points
        and planes that represent reflective symmetries.
        :param data_source_path: Path to folder that contains the points and symmetries.
        :param transform: Transform applied to dataset item.
        """
        self.data_source_path = Path(data_source_path)
        self.transform = transform
        #self.length = len(os.listdir(self.data_source_path)) // 2
        self.has_ground_truth = has_ground_truth
        self.debug = debug

        if self.debug:
           print(f'Searching xz-compressed point cloud files in {self.data_source_path}...')
        self.flist  = list(self.data_source_path.rglob(f'*/*.xz'))
        self.length = len(self.flist)
        if self.debug:
            print(f'{self.data_source_path.name}: found {self.length} files:\n{self.flist[:5]}\n{self.flist[-5:]}\n')

    def fname_from_idx(self, idx: int) -> str:
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

        points = None
        with lzma.open(fname, 'rb') as fhandle:
            points = torch.tensor(np.loadtxt(fhandle))

        if self.debug:
           torch.set_printoptions(linewidth=200)
           torch.set_printoptions(precision=3)
           torch.set_printoptions(sci_mode=False)
           print(f'[{idx}]: {points.shape = }\n{points = }')

        return points

    def read_planes(self, idx: int) -> torch.Tensor:
        """
        Read symmetry planes from file with its first line being the number of symmetry planes
        and the rest being the symmetry planes.
        :param idx: The idx of the syms to reads
        :return: A tensor of planes represented by their normals and points. N x 6 where
        N is the amount of planes and 6 because the first 3 elements
        are the normal and the last 3 are the point.
        """
        _, sym_fname = self.fname_from_idx(idx)
        with open(sym_fname) as f:
            n_planes = int(f.readline().strip())
            #converters = {1: lambda s: [0 if s == 'plane' else 1]}
            #sym_planes = torch.tensor(np.loadtxt(f, converters=converters, usecols=range(1,7)))
            #sym_planes = torch.tensor(np.loadtxt(f, usecols=range(1,8)))
            if self.debug:
                print(f'Reading CSV dataframe with filename:\n{Path(sym_fname).name}')
            #df = pd.read_csv(f, sep=' ', header=None, usecols=range(0,8), names=['type', 'nx', 'ny', 'nz', 'cx', 'cy', 'cz', 'theta']).fillna(-1) # 'ϑ'
            try:
                '''
                TODO: restore me and fix me to be able to load the old dataset!!!
                if self.debug:
                    print(f'Reading CSV dataframe with theta column')
                df = pd.read_csv(f, sep=' ', header=None, names=['type', 'nx', 'ny', 'nz', 'cx', 'cy', 'cz', 'theta']).fillna(-1) # 'ϑ'
                '''
                if self.debug:
                    print(f'Reading CSV dataframe with the old dataset format (normals+points only)')
                df = pd.read_csv(f, sep=' ', header=None, names=['nx', 'ny', 'nz', 'cx', 'cy', 'cz']).fillna(-1)
                #df['type'] = df['type'].astype('int32')
                df['type'] = 'plane'
            except pd.errors.ParserError:
                if self.debug:
                    print(f'Re-reading CSV dataframe without theta column')
                # NOTE here that we read the file directly, so we must throw away the first row
                df = pd.read_csv(sym_fname, sep=' ', header=None, names=['type', 'nx', 'ny', 'nz', 'cx', 'cy', 'cz']).fillna(-1) # 'ϑ'
                df = df.iloc[1:]
            if self.debug:
                print(f'Read dataframe:\n{df}')
            df['type'] = np.where(df['type'] == 'plane', 0, 1)
            if self.debug:
                print(f'Converted dataframe:\n{df}')
            ''' ------------------------------------------------------------------------------------- '''
            ''' ------------------------------------------------------------------------------------- '''
            ''' ------------------------------------------------------------------------------------- '''
            
            # TODO: enable these lines throw away some information present in the new version of the dataset
            dfbackup = df.copy()
            if (df['type'] == 1).any() == True:          # there is an axial symmetry
                df = df[df['type'] != 1]                 # throw away axial symmetry rows
                df = df.drop('theta', axis=1)            # throw away last column == angles
                n_planes = n_planes - 1                  # decrease the number of reported symmetries
            
            if len(df) == 0 or n_planes == 0:            # there was only an axial symmetry, nothing else
                df = dfbackup                            # restore original dataframe
            ''' ------------------------------------------------------------------------------------- '''
            ''' ------------------------------------------------------------------------------------- '''
            ''' ------------------------------------------------------------------------------------- '''
            df = df.drop('type', axis=1)                 # throw away 1st column  == plane/axis flags
            if self.debug:
                print(f'Resized dataframe:\n{df}')

            sym_planes = torch.tensor(df.values)
            if self.debug:
                print(f'Exported dataframe to torch.tensor with shape: {sym_planes.shape}\n{sym_planes}')
        #if n_planes == 1:
        #    sym_planes = sym_planes.unsqueeze(0)

        if self.debug:
           print(f'[{idx}]: {n_planes = }\n{sym_planes}')

        return sym_planes

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> (int, torch.Tensor, Optional[torch.Tensor], List[Shrec2023Transform]):
        points = self.read_points(idx)
        planes = None

        if self.has_ground_truth:
            planes = self.read_planes(idx)

        if self.transform is not None:
            idx, points, planes = self.transform(idx, points, planes)

        transform_used = copy.deepcopy(self.transform)
        return idx, points.float(), planes.float(), transform_used


scaler = UnitSphereNormalization()
sampler = RandomSampler(sample_size=1024, keep_copy=False)
# default_transform = ComposeTransform([scaler, sampler])
default_transform = scaler


class SymmetryDataModule(lightning.LightningDataModule):
    def __init__(
            self,
            #train_data_path  : str = "/path/to/train_data",
            #valid_data_path  : str = "/path/to/valid_data",
            #test_data_path   : str = "/path/to/test_data",
            dataset_path: str = "/path/to/dataset",
            predict_data_path: str = "/path/to/predict_data",
            does_predict_has_ground_truths: bool = False,
            batch_size: int = 2,
            transform: Optional[Shrec2023Transform] = None,
            collate_function = None,
            #validation_percentage: float = 0.1,
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
        self.collate_function = default_symmetry_dataset_collate_fn_list_sym if collate_function is None else collate_function
        #self.validation_percentage = validation_percentage
        self.shuffle = shuffle
        self.n_workers = n_workers

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = SymmetryDataset(
                data_source_path=Path(self.dataset_path) / 'train',
                transform=self.transform,
                has_ground_truth=True,
                debug=True
            )
            self.valid_dataset = SymmetryDataset(
                data_source_path=Path(self.dataset_path) / 'valid',
                transform=self.transform,
                has_ground_truth=True,
                debug=True
            )

            '''
            proportions = [1 - self.validation_percentage, self.validation_percentage]
            lengths = [int(p * len(dataset_full)) for p in proportions]
            lengths[-1] = len(dataset_full) - sum(lengths[:-1])

            self.train_dataset, self.validation_dataset = random_split(
                dataset_full, lengths
            )
            '''

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

    #DATA_PATH = "/data/shrec_2023/benchmark-train"
    DATA_PATH = "/mnt/btrfs-big/dataset/geometric-primitives-classification/symmetry-datasets/sym-10k-xz-split-class-noparallel"

    scaler = UnitSphereNormalization()
    sampler = RandomSampler(sample_size=3, keep_copy=True)
    default_transform = ComposeTransform([scaler, sampler])

    train_dataset = SymmetryDataset(Path(DATA_PATH) / 'train', default_transform)
    valid_dataset = SymmetryDataset(Path(DATA_PATH) / 'valid', default_transform)
    test_dataset  = SymmetryDataset(Path(DATA_PATH) / 'test' , default_transform)

    example_idx, example_points, example_syms, example_tr = train_dataset[0]
    print("transformed", example_idx, example_points[0, :], example_syms[0, :], example_tr)
    ridx, rpoints, rsyms = example_tr.inverse_transform(example_idx, example_points, example_syms)
    print("inverse_transform", ridx, rpoints[0, :], rsyms[0, :])
    ridx, rpoints, rsyms = example_tr.transform(ridx, rpoints, rsyms)
    print("transformed 2", ridx, rpoints[0, :], rsyms[0, :])

    datamodule = SymmetryDataModule(
        dataset_path=DATA_PATH,
        #train_data_path=DATA_PATH,
        #valid_data_path=DATA_PATH,
        #test_data_path=DATA_PATH,
        predict_data_path=DATA_PATH,
        does_predict_has_ground_truths=True,
        batch_size=100,
        transform=default_transform,
        collate_function=default_symmetry_dataset_collate_fn_list_sym,
        #collate_function=default_symmetry_dataset_collate_fn,
        #validation_percentage=0.1,
        shuffle=True,
        n_workers=1,
    )
    datamodule.setup("fit")

    train_dataloader = datamodule.train_dataloader()
    print(f'The training dataloader has: {len(train_dataloader)} batches')

    batch = next(iter(train_dataloader))

    for x in batch:
        if isinstance(x, torch.Tensor):
            print(x.shape)
        else:
            print(len(x))
