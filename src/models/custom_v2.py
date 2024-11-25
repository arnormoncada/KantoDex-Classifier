import torch
from torch import nn


class Bottleneck(nn.Module):
    """
    Bottleneck block.

    This block consists of three convolution layers:
    1. 1x1 convolution for dimensionality reduction
    2. 3x3 convolution for feature processing
    3. 1x1 convolution for dimensionality expansion

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of intermediate channels (expanded by expansion factor)
        stride (int, optional): Stride for the 3x3 convolution. Defaults to 1
        downsample (Optional[nn.Module], optional): Downsample layer for residual connection.
            Defaults to None

    """

    expansion: int = 4  # Expansion factor for output channels

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        # 1x1 convolution for dimensionality reduction
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 3x3 convolution for feature processing
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 1x1 convolution for dimensionality expansion
        self.conv3 = nn.Conv2d(
            out_channels,
            out_channels * self.expansion,
            kernel_size=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Bottleneck block.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor after bottleneck operations and residual connection

        """
        identity = x  # Residual connection

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out  # noqa: RET504


class DeepCNN(nn.Module):
    """
    Deep CNN model with Bottleneck blocks for classification.

    Args:
        num_classes (int, optional): Number of output classes. Defaults to 151.
        dropout (float, optional): Dropout probability. Defaults to 0.2.

    """

    def __init__(self, num_classes: int = 151, dropout: float = 0.2) -> None:
        super().__init__()
        self.in_channels = 64

        # Stem: initial convolution and pooling
        self.stem = nn.Sequential(
            nn.Conv2d(
                3,
                self.in_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),  # Output: (64, H/2, W/2)
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # Output: (64, H/4, W/4)
        )

        # Build layers with Bottleneck blocks
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 5, stride=2)
        self.layer3 = self._make_layer(256, 7, stride=2)
        self.layer4 = self._make_layer(512, 4, stride=2)

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(
        self,
        out_channels: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        """
        Create a sequential layer composed of Bottleneck blocks.

        Args:
            out_channels (int): Number of intermediate channels
            blocks (int): Number of Bottleneck blocks
            stride (int, optional): Stride for the first block. Defaults to 1.

        Returns:
            nn.Sequential: Sequential container of Bottleneck blocks

        """
        downsample: nn.Sequential | None = None
        if stride != 1 or self.in_channels != out_channels * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * Bottleneck.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * Bottleneck.expansion),
            )

        layers = []
        layers.append(
            Bottleneck(
                in_channels=self.in_channels,
                out_channels=out_channels,
                stride=stride,
                downsample=downsample,
            ),
        )
        self.in_channels = out_channels * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(
                Bottleneck(
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                ),
            )

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DeepCNN model.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output logits for classification

        """
        x = self.stem(x)  # Shape: (batch, 64, H/4, W/4)
        x = self.layer1(x)  # Shape: (batch, 256, H/4, W/4)
        x = self.layer2(x)  # Shape: (batch, 512, H/8, W/8)
        x = self.layer3(x)  # Shape: (batch, 1024, H/16, W/16)
        x = self.layer4(x)  # Shape: (batch, 2048, H/32, W/32)

        x = self.avgpool(x)  # Shape: (batch, 2048, 1, 1)
        x = torch.flatten(x, 1)  # Shape: (batch, 2048)
        x = self.dropout(x)  # Apply dropout
        x = self.fc(x)  # Shape: (batch, num_classes)

        return x  # noqa: RET504

    def _initialize_weights(self) -> None:
        """
        Initialize model weights.

        Initialize model weights using Kaiming normalization for conv layers
        and constant initialization for batch norm and linear layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# Example usage:
if __name__ == "__main__":
    model = DeepCNN(num_classes=151)
    print(model)

    # Test with a random input
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(output.shape)  # Expected: (1, 151)
