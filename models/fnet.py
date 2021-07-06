import torch
import torch.nn as nn
import torch.nn.functional as F


class FNet(nn.Module):
    """ Optical flow estimation network
    """

    def __init__(self, in_nc):
        super(FNet, self).__init__()

        self.encoder1 = nn.Sequential(
            nn.Conv2d(2 * in_nc, 32, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2))

        self.encoder2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2))

        self.encoder3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2))

        self.decoder1 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True))

        self.decoder2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True))

        self.decoder3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True))

        self.flow = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 2, 3, 1, 1, bias=True))

    def forward(self, x1, x2):
        """ Compute optical flow from x1 to x2
        """

        out = self.encoder1(torch.cat([x1, x2], dim=1))
        out = self.encoder2(out)
        out = self.encoder3(out)
        out = F.interpolate(
            self.decoder1(out), scale_factor=2, mode='bilinear', align_corners=False)
        out = F.interpolate(
            self.decoder2(out), scale_factor=2, mode='bilinear', align_corners=False)
        out = F.interpolate(
            self.decoder3(out), scale_factor=2, mode='bilinear', align_corners=False)
        out = torch.tanh(self.flow(out)) * 24  # 24 is the max velocity

        return out
