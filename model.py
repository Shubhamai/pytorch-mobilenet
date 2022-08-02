"""A from-scratch implementation of original MobileNet paper ( for educational purposes ).

Paper
    MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications - https://arxiv.org/abs/1704.04861

author : shubham.aiengineer@gmail.com
"""

import torch
import torch.nn.functional as F
# Importing Libraries
from torch import nn
from torchsummary import summary


class DepthwiseConvBlock(nn.Module):
    """Depthwise seperable with pointwise convolution with relu and batchnorm respectively.

    Attributes:
        in_channels: Input channels for depthwise convolution
        out_channels: Output channels for pointwise convolution
        stride: Stride paramemeter for depthwise convolution
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        """Initialize parameters."""
        super().__init__()

        # Depthwise conv
        self.depthwise_conv = nn.Conv2d(
            in_channels,
            in_channels,
            (3, 3),
            stride=stride,
            padding=1,
            groups=in_channels,
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU()

        # Pointwise conv
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, (1, 1))
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        """Perform forward pass."""

        x = self.depthwise_conv(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pointwise_conv(x)
        x = self.bn2(x)
        x = self.relu2(x)

        return x


class MobileNetV1(nn.Module):
    """Constructs MobileNetV1 architecture

    Attributes:
        n_classes: Number of output neuron in last layer
        alpha:
    """

    def __init__(self, n_classes: int = 1000, alpha: float = 1.0):
        """Initialize parameters."""
        super().__init__()

        self.model = nn.Sequential(nn.Conv2d(3, 32, (3, 3), stride=2, padding=1))

        for layer in config:
            self.model.append(DepthwiseConvBlock(layer[0], layer[1], stride=layer[2]))

        self.model.append(nn.AvgPool2d(7))
        self.model.append(nn.Flatten())
        self.model.append(nn.Linear(1024, n_classes))
        self.model.append(nn.Softmax())

    def forward(self, x):
        """Perform forward pass."""
        x = self.model(x)

        return x


config = (
    (32, 64, 1),
    (64, 128, 2),
    (128, 128, 1),
    (128, 256, 2),
    (256, 256, 1),
    (256, 512, 2),
    (512, 512, 1),
    (512, 512, 1),
    (512, 512, 1),
    (512, 512, 1),
    (512, 512, 1),
    (512, 1024, 2),
    (1024, 1024, 1),
)

if __name__ == "__main__":

    # Generating Sample image
    image_size = (1, 3, 224, 224)
    image = torch.rand(*image_size)

    # Model
    mobilenet_v1 = MobileNetV1()

    summary(mobilenet_v1, input_size=image_size)

    out = mobilenet_v1(image)
    print("Output shape : ", out.shape)
