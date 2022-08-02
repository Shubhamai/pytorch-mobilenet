"""A from-scratch implementation of MobileNetV2 paper ( for educational purposes ).

Paper
    MobileNetV2: Inverted Residuals and Linear Bottlenecks - https://arxiv.org/abs/1801.04381

author : shubham.aiengineer@gmail.com
"""


import torch
from torch import nn
from torchsummary import summary

# The configuration of MobileNet with depth multiplier set to 1.
config = ((32, 16, 1),)  # input channels, output channels, stride


class MBConvBlock(nn.Module):
    """Constructs a bottleneck residual block"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, 1))
        self.relu1 = nn.ReLU6()
        self.depthwise_conv = nn.Conv2d(
            out_channels,
            out_channels,
            (3, 3),
            stride=stride,
            padding=1,
            groups=out_channels,
        )
        self.relu2 = nn.ReLU6()
        self.conv2 = nn.Conv2d(out_channels, in_channels, (1, 1))

    def forward(self, x):
        """Perform forward pass."""

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.depthwise_conv(x)
        x = self.relu2(x)
        x = self.conv2(x)

        return x


class MobileNetV2(nn.Module):
    """Constructs MobileNetV2 architecture"""

    def __init__(
        self,
        n_classes: int = 1000,
        input_channel: int = 3,
        depth_multiplier: float = 1.0,
    ):
        """Attributes:
        `n_classes`: An integer count of output neuron in last layer.
        `input_channel`: An integer value input channels in first conv layer
        `depth_multiplier` (0, 1] : A float value indicating network width multiplier ( width scaling ). Suggested Values - 0.25, 0.5, 0.75, 1.
        """

        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(
                input_channel, int(32 * depth_multiplier), (3, 3), stride=2, padding=1
            )
        )

        for in_channels, out_channels, stride in config:
            self.model.append(
                MBConvBlock(
                    int(in_channels * depth_multiplier),
                    int(out_channels * depth_multiplier),
                    stride,
                )
            )

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
    )

    out = mobilenet_v2(image)
    print("Output shape : ", out.shape)
