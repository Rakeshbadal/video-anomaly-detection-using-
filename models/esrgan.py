import torch.nn as nn

class ESRGANGenerator(nn.Module):
    def __init__(self):
        super(ESRGANGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            *[nn.Conv2d(64, 64, 3, 1, 1) for _ in range(5)],
            nn.Conv2d(64, 3, 3, 1, 1)
        )

    def forward(self, x):
        return self.model(x)
