#!/usr/bin/env python3
import torch
from lightning.pytorch.cli import LightningCLI
from src.dataset.SymDataModule import SymDataModule
from src.model.LightingCenterNNormalsNet import LightingCenterNNormalsNet


def cli_main():
    torch.set_float32_matmul_precision('medium')
    cli = LightningCLI(LightingCenterNNormalsNet, SymDataModule)


if __name__ == "__main__":
    cli_main()
