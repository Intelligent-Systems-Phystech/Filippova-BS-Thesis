from .base import Backbone
from .unet_blocks import InConvBlock, DownBlock, UpBlock, OutConvBlock

from torch.nn import Softmax
import torch

class UNet(Backbone):
    def __init__(self, channels):
        super().__init__()

        assert len(channels) == 7
        self.channels = channels
        self.inc = InConvBlock(channels[0], channels[1])
        self.down1 = DownBlock(channels[1], channels[2])
        self.down2 = DownBlock(channels[2], channels[3])
        self.down3 = DownBlock(channels[3], channels[4])
        self.down4 = DownBlock(channels[4], channels[4])
        self.up1 = UpBlock(channels[5], channels[3])
        self.up2 = UpBlock(channels[4], channels[2])
        self.up3 = UpBlock(channels[3], channels[1])
        self.up4 = UpBlock(channels[2], channels[1])
        self.outc = OutConvBlock(channels[1], channels[6])
        self.softmax = Softmax(dim=-1)
    
    def forward(self, x):

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        out = x.permute(0, 2, 1)
        out = self.softmax(out)
        
        return out
