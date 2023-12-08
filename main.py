from lightning.pytorch.cli import LightningCLI
from src.dataset.shrec2023 import SymmetryDataModule
from src.model.my_net.network import LightingMyNet
import torch


def cli_main():
    torch.set_float32_matmul_precision('high')
    cli = LightningCLI(LightingMyNet, SymmetryDataModule)


if __name__ == "__main__":
    cli_main()
