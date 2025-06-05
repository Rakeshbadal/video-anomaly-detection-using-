import torch
import torch.nn as nn

class ConvLSTM3D(nn.Module):
    def __init__(self, input_channels=3, hidden_dim=64):
        super(ConvLSTM3D, self).__init__()
        self.conv1 = nn.Conv3d(input_channels, hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv3d(hidden_dim, input_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x
