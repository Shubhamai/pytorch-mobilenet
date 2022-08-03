"""A from-scratch implementation of MobileNetV2 paper ( for educational purposes ).

Paper
    MobileNetV2: Inverted Residuals and Linear Bottlenecks - https://arxiv.org/abs/1801.04381

author : shubham.aiengineer@gmail.com
"""


import torch
from torch import nn
from torchsummary import summary

# The configuration of MobileNet with depth multiplier set to 1.

# input channels, output channels, stride, expansion factor, repeat
config = (
    (32, 16, 1, 1, 1),
    (16, 24, 2, 6, 2),
    (24, 32, 2, 6, 3),
    (32, 64, 2, 6, 4),
    (64, 96, 1, 6, 3),
    (96, 160, 2, 6, 3),
    (160, 320, 1, 6, 1),
)


class ConvBatchReLUBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: list,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        bias:bool=False
    ):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels,kernel_size, stride=stride, padding=padding, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6()

    def forward(self, x):
        """Perform forward pass."""

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class InverseResidualBlock(nn.Module):
    """Constructs a inverse residual block with depthwise seperable convolution"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion_factor: int = 6,
        stride: int = 1,
    ):

        """Attributes:
        `in_channels`: Integer indicating input channels
        `out_channels`: Integer indicating output channels
        `stride`: Integer indicating stride paramemeter for depthwise convolution
        `expansion_factor` : Calculating the input & output channel for depthwise convolution by multiplying the expansion factor with input channels"""
        super().__init__()

        hidden_channels = in_channels * expansion_factor
        self.residual = in_channels == out_channels and stride == 1

        self.conv1 = nn.Conv2d(in_channels, hidden_channels, (1, 1))
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.relu1 = nn.ReLU6()
        self.depthwise_conv = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            (3, 3),
            stride=stride,
            padding=1,
            groups=hidden_channels,
        )
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.relu2 = nn.ReLU6()
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, (1, 1))

    def forward(self, x):
        """Perform forward pass."""

        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv2(x)

        if self.residual:
            x = torch.add(x, identity)

        return x


class MobileNetV2(nn.Module):
    """Constructs MobileNetV2 architecture"""

    def __init__(
        self,
        n_classes: int = 1000,
        input_channel: int = 3,
        dropout: float = 0.2,
    ):
        """Attributes:
        `n_classes`: An integer count of output neuron in last layer.
        `input_channel`: An integer value input channels in first conv layer
        """

        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(input_channel, 32, (3, 3), stride=2, padding=1)
        )
        self.model.append(nn.BatchNorm2d(32))
        self.model.append(nn.ReLU6())

        for in_channels, out_channels, stride, expansion_factor, repeat in config:
            for _ in range(repeat):
                self.model.append(
                    InverseResidualBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        expansion_factor=expansion_factor,
                        stride=stride,
                    )
                )
                in_channels = out_channels
                stride = 1

        self.model.append(nn.Conv2d(320, 1028, (1, 1)))
        self.model.append(nn.BatchNorm2d(1028))
        self.model.append(nn.ReLU6())
        self.model.append(nn.AdaptiveAvgPool2d(1))
        self.model.append(nn.Flatten())
        self.model.append(nn.Dropout(dropout))
        self.model.append(nn.Linear(1028, n_classes))

    def forward(self, x):
        """Perform forward pass."""

        x = self.model(x)

        return x


if __name__ == "__main__":

    # Generating Sample image
    image_size = (1, 3, 224, 224)
    image = torch.rand(*image_size)

    # Model
    mobilenet_v2 = MobileNetV2()

    summary(
        mobilenet_v2,
        input_data=image,
        col_names=["input_size", "output_size", "num_params"],
        device="cpu",
        depth=2,
    )

    out = mobilenet_v2(image)
    print("Output shape : ", out.shape)
