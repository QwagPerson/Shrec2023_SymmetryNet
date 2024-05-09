from torch import nn


class DiscreteRotationalSymmetryLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss1 = nn.CrossEntropyLoss()
        self.loss2 = nn.KLDivLoss()
        self.loss3 = nn.MSE()
        self.loss4 = nn.MSE()

        self.a = nn.Parameter(0.25)
        self.b = nn.Parameter(0.25)
        self.c = nn.Parameter(0.25)
        self.d = nn.Parameter(0.25)

    def forward(self, x):
        return self.loss1(x) * self.a + self.loss2(x) * self.b + self.loss3(x) * self.c + self.loss4(x) * self.d