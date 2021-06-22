import torch

from torch import nn
from .base import Backbone


class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class InConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConvBlock(in_channels, out_channels)

    def forward(self, x):
        x = self.conv(x)
        return x


class OutConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.conv(x)
        return x


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.mpconv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, linear=False):
        super().__init__()

        #  would be a nice idea if the upsampling could be learned too
        if linear:
            self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        else:
            self.up = nn.ConvTranspose1d(in_channels // 2, in_channels // 2, 2, stride=2)


        self.conv = DoubleConvBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        assert x1.dim() == 3
        diffY = x2.size()[-1] - x1.size()[-1]
        x1 = nn.functional.pad(x1, (diffY // 2, diffY - diffY // 2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

