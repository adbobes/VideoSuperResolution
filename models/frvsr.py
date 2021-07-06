import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


from utils.upsample import BicubicUpsample


class ResidualBlock(nn.Module):
    def __init__(self, channel=64):
        super(ResidualBlock, self).__init__()

        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(channel, channel, 3, 1, 1, bias=True)),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 3, 1, 1, bias=True))
        self.relu_zero = 0

    def forward(self, input):
        out = self.conv(input) + input
            
        return out


class RDN(nn.Module):
    def __init__(self, channel=128):
        super(RDN, self).__init__()

        # input
        self.conv_in = nn.Sequential(
            spectral_norm(nn.Conv2d(51, channel, 3, 1, 1, bias=True)),
            nn.ReLU(inplace=True)
        )

        # resblock
        self.resblocks = nn.Sequential(*[ResidualBlock(channel) for _ in range(20)])

        # upsampling

        self.conv_up = nn.Sequential(
            nn.ConvTranspose2d(channel, channel, 3, 2, 1, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channel, channel, 3, 2, 1, 1, bias=True),
            nn.ReLU(inplace=True))

        # output
        self.conv_out = nn.Conv2d(channel, 3, 3, 1, 1, bias=True)


        self.upsample_func = BicubicUpsample(4)

    def forward(self, lr_curr, lr_in):
        inp = self.conv_in(torch.cat([lr_curr, lr_in], dim=1))
        res = self.resblocks(inp)
        up = self.conv_up(res)
        out = self.conv_out(up)
        out = out + self.upsample_func(lr_curr)

        return out

