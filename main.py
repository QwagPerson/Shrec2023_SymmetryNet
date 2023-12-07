from lightning.pytorch.cli import LightningCLI
from src.model.network import LightingSymmetryNet
from src.dataset.shrec2023 import SymmetryDataModule
import torch


def cli_main():
    torch.set_float32_matmul_precision('high')
    cli = LightningCLI(LightingSymmetryNet, SymmetryDataModule)


if __name__ == "__main__":
    cli_main()
