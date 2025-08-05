import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def gabor_kernel(kernel_size, theta, lam, sigma, gamma, phi, device=None, dtype=None):
    """
    Generate a 2D Gabor kernel.
    Args:
        kernel_size: size of the kernel (int)
        theta: orientation (radians)
        lam: wavelength (lambda)
        sigma: standard deviation of gaussian
        gamma: spatial aspect ratio
        phi: phase offset
    Returns:
        torch.Tensor of shape (kernel_size, kernel_size)
    """
    # Create coordinate grid
    half = (kernel_size - 1) / 2.
    y, x = torch.meshgrid(
        [torch.linspace(-half, half, steps=kernel_size),
         torch.linspace(-half, half, steps=kernel_size)]
    )
    # Rotation
    x_theta = x * math.cos(theta) + y * math.sin(theta)
    y_theta = -x * math.sin(theta) + y * math.cos(theta)
    # Gabor formula
    gb = torch.exp(
        -0.5 * (x_theta**2 + (gamma**2) * y_theta**2) / (sigma**2)
    ) * torch.cos(
        2 * math.pi * x_theta / lam + phi
    )
    return gb.to(device=device, dtype=dtype)


class GaborConv(nn.Module):
    """
    Gabor-modulated convolutional layer.
    Applies multiple Gabor filters at different orientations,
    concatenates the results, and applies a pointwise convolution.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=11,
                 orientations=None,
                 lam=10.0,
                 sigma=4.0,
                 gamma=0.5,
                 phi=0.0,
                 stride=1,
                 padding=None,
                 bias=True):
        super(GaborConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.orientations = orientations or [i * math.pi / 8 for i in range(8)]
        self.lam = lam
        self.sigma = sigma
        self.gamma = gamma
        self.phi = phi
        self.stride = stride
        self.padding = padding if padding is not None else kernel_size // 2

        # pointwise conv to mix orientation responses
        self.pointwise = nn.Conv2d(
            in_channels * len(self.orientations),
            out_channels,
            kernel_size=1,
            bias=bias
        )

    def forward(self, x):
        # x: (B, C, H, W)
        outputs = []
        B, C, H, W = x.shape
        device = x.device
        dtype = x.dtype

        for theta in self.orientations:
            # generate Gabor kernel
            k = gabor_kernel(
                self.kernel_size, theta,
                self.lam, self.sigma,
                self.gamma, self.phi,
                device=device,
                dtype=dtype
            )  # (K, K)
            # expand to depthwise conv weight: (C,1,K,K)
            weight = k.unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1)
            # depthwise convolution: groups=C
            gi = F.conv2d(
                x, weight,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                groups=C
            )  # (B, C, H, W)
            outputs.append(gi)

        # concatenate along channel dim: (B, C*O, H, W)
        cat = torch.cat(outputs, dim=1)
        # pointwise to mix orientations: (B, out_channels, H, W)
        y = self.pointwise(cat)
        return y

