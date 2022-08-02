"""A from-scratch implementation of original MobileNet paper ( for educational purposes ).

Paper
    MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications - https://arxiv.org/abs/1704.04861

author : shubham.aiengineer@gmail.com
"""

# Importing Libraries
import torch
from torch import nn
from torchsummary import summary

# The configuration of MobileNet with depth multiplier set to 1.
# input channels, output channels, stride
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


class DepthwiseSepConvBlock(nn.Module):
    """Constructs Depthwise seperable with pointwise convolution with relu and batchnorm respectively."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_relu6: bool = True,
    ):
        """Attributes:
        `in_channels`: Integer indicating input channels for depthwise convolution
        `out_channels`: Integer indicating output channels for pointwise convolution
        `stride`: Integer indicating stride paramemeter for depthwise convolution
        `use_relu6` : A boolean value indicating whether to use standard ReLU or ReLU6 for depthwise separable convolution block"""

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

        if use_relu6:
            self.relu1 = nn.ReLU6()
        else:
            self.relu1 = nn.ReLU()

        # Pointwise conv
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, (1, 1))
        self.bn2 = nn.BatchNorm2d(out_channels)

        if use_relu6:
            self.relu2 = nn.ReLU6()
        else:
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
    """Constructs MobileNetV1 architecture"""

    def __init__(
        self,
        n_classes: int = 1000,
        input_channel: int = 3,
        depth_multiplier: float = 1.0,
        use_relu6: bool = True,
    ):
        """Attributes:
        `n_classes`: An integer count of output neuron in last layer.
        `input_channel`: An integer value input channels in first conv layer
        `depth_multiplier` (0, 1] : A float value indicating network width multiplier ( width scaling ). Suggested Values - 0.25, 0.5, 0.75, 1.
        `use_relu6` : A boolean value indicating whether to use standard ReLU or ReLU6 for depthwise separable convolution block"""
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(
                input_channel, int(32 * depth_multiplier), (3, 3), stride=2, padding=1
            )
        )

        for in_channels, out_channels, stride in config:
            self.model.append(
                DepthwiseSepConvBlock(
                    int(in_channels * depth_multiplier),
                    int(out_channels * depth_multiplier),
                    stride,
                    use_relu6=use_relu6,
                )
            )

        self.model.append(nn.AdaptiveAvgPool2d(1))
        self.model.append(nn.Flatten())
        self.model.append(nn.Linear(int(1024 * depth_multiplier), n_classes))
        # self.model.append(nn.Softmax())

    def forward(self, x):
        """Perform forward pass."""
        x = self.model(x)

        return x


if __name__ == "__main__":

    # Generating Sample image
    image_size = (1, 3, 224, 224)
    image = torch.rand(*image_size)

    # Model
    mobilenet_v1 = MobileNetV1(depth_multiplier=1)

    summary(
        mobilenet_v1,
        input_data=image,
        col_names=["input_size", "output_size", "num_params"],
        device="cpu",
    )

    out = mobilenet_v1(image)
    print("Output shape : ", out.shape)
