from torch import nn


class DensePredictionDecoder(nn.Module):
    """
    Transforms the output of a PointNetEncoder into
    B x 512 x S -> B x 6 x S
    """

    def __init__(
            self,
            batch_size: int,
            sample_size: int,
    ):
        super(PointNetEncoder, self).__init__()
        self.batch_size = batch_size
        self.sample_size = sample_size

        self.input_transform = TNet(in_dim=3, k=3)
        self.shared_mlps = torch.nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.feature_transform = TNet(in_dim=64, k=64)
        self.shared_mlps_2 = torch.nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
        )

        self.swp = Swp1d(batch_size, sample_size, 1)