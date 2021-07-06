import torch
import torch.nn as nn
import torch.nn.functional as F


class BicubicUpsample(nn.Module):
    """ Bicubic upsampling function with similar behavior to that in TecoGAN-Tensorflow
        Note:
            This function is different from torch.nn.functional.interpolate and matlab's imresize
            in terms of the bicubic kernel and the sampling strategy.
        References:
            http://verona.fi-p.unam.mx/boris/practicas/CubConvInterp.pdf
            https://stackoverflow.com/questions/26823140/imresize-trying-to-understand-the-bicubic-interpolation
    """

    def __init__(self, scale_factor, a=-0.75):
        super(BicubicUpsample, self).__init__()

        # calculate weights (according to Eq.(6) in the reference paper)
        cubic = torch.FloatTensor([
            [0, a, -2 * a, a],
            [1, 0, -(a + 3), a + 2],
            [0, -a, (2 * a + 3), -(a + 2)],
            [0, 0, a, -a]
        ])

        kernels = [
            torch.matmul(cubic, torch.FloatTensor([1, s, s ** 2, s ** 3]))
            for s in [1.0 * d / scale_factor for d in range(scale_factor)]
        ]  # s = x - floor(x)

        # register parameters
        self.scale_factor = scale_factor
        self.register_buffer('kernels', torch.stack(kernels))  # size: (f, 4)

    def forward(self, input):
        n, c, h, w = input.size()
        f = self.scale_factor

        # merge n&c
        input = input.reshape(n * c, 1, h, w)

        # pad input (left, right, top, bottom)
        input = F.pad(input, (1, 2, 1, 2), mode='replicate')

        # calculate output (vertical expansion)
        kernel_h = self.kernels.view(f, 1, 4, 1).to(input.device)
        output = F.conv2d(input, kernel_h, stride=1, padding=0)
        output = output.permute(0, 2, 1, 3).reshape(n * c, 1, f * h, w + 3)

        # calculate output (horizontal expansion)
        kernel_w = self.kernels.view(f, 1, 1, 4).to(input.device)
        output = F.conv2d(output, kernel_w, stride=1, padding=0)
        output = output.permute(0, 2, 3, 1).reshape(n * c, 1, f * h, f * w)

        # split n&c
        output = output.reshape(n, c, f * h, f * w)

        return output
