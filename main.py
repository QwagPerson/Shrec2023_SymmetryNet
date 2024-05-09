#!/usr/bin/env python3
from lightning.pytorch.cli import LightningCLI
from src.dataset.SymmetryDataModule import SymmetryDataModule
from src.model.LightingCenterNNormalsNet import LightingCenterNNormalsNet


def cli_main():
    cli = LightningCLI(LightingCenterNNormalsNet, SymmetryDataModule)


if __name__ == "__main__":
    cli_main()
