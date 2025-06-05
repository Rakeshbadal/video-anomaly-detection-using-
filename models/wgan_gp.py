import torch.nn as nn

class WGANCritic(nn.Module):
    def __init__(self, input_channels=3):
        super(WGANCritic, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.Flatten(),
            nn.Linear(256 * 16 * 16, 1)  # Adjust size to input image
        )

    def forward(self, x):
        return self.model(x)
