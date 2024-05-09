#!/usr/bin/env python3
from lightning.pytorch.cli import LightningCLI
from src.dataset.SymDataModule import SymDataModule
from src.model.LightingCenterNNormalsNet import LightingCenterNNormalsNet


def cli_main():
    cli = LightningCLI(LightingCenterNNormalsNet, SymDataModule)


if __name__ == "__main__":
    cli_main()
