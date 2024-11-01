import math

import torch
import torch.nn.functional as F
from torch import nn


class PositionalEncoding(nn.Module):
    """Positional Encoding module adds positional information to the input embeddings."""

    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)  # Shape: (max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # Shape: (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim),
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, embed_dim)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_length, embed_dim)

        Returns:
            torch.Tensor: Tensor with positional encoding added.

        """
        x = x + self.pe[:, : x.size(1), :]
        return x  # noqa: RET504


class SEBlock(nn.Module):
    """Squeeze-and-Excitation (SE) Block to recalibrate channel-wise feature responses."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        batch, channels, _, _ = x.size()
        # Squeeze: Global Average Pooling
        y = F.adaptive_avg_pool2d(x, 1).view(batch, channels)
        # Excitation: FC -> ReLU -> FC -> Sigmoid
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(batch, channels, 1, 1)
        # Scale
        return x * y.expand_as(x)


class ResidualConvBlock(nn.Module):
    """Residual Convolutional Block with SE Block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        reduction: int = 16,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels, reduction)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out  # noqa: RET504


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention (MHSA) Block."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        # x shape: (batch, seq_length, embed_dim)
        x_permuted = x.permute(
            1,
            0,
            2,
        )  # Convert to (seq_length, batch, embed_dim) for nn.MultiheadAttention
        attn_output, _ = self.attention(x_permuted, x_permuted, x_permuted)
        attn_output = attn_output.permute(1, 0, 2)  # Back to (batch, seq_length, embed_dim)
        x = self.norm(x + self.dropout(attn_output))
        return x  # noqa: RET504


class DropBlock2D(nn.Module):
    """DropBlock Regularization for 2D feature maps."""

    def __init__(self, block_size: int, drop_prob: float) -> None:
        super().__init__()
        self.block_size = block_size
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        if not self.training or self.drop_prob == 0.0:
            return x

        # Ensure input is 4D
        if x.dim() != 4:  # noqa: PLR2004
            msg = f"DropBlock2D expects 4D input (batch, channels, height, width), but got {x.dim()}D input."  # noqa: E501
            raise ValueError(
                msg,
            )

        # Calculate gamma based on drop_prob and block_size
        gamma = self.drop_prob / (self.block_size**2)

        # Sample mask
        # Shape: (batch_size, channels, height, width)  # noqa: ERA001
        mask = torch.bernoulli(torch.ones_like(x) * gamma)

        # Apply max pooling to create contiguous blocks
        mask = F.max_pool2d(
            mask,
            kernel_size=self.block_size,
            stride=1,
            padding=self.block_size // 2,
        )

        # Invert mask: 1 means keep, 0 means drop
        mask = 1 - mask

        # Ensure mask has the same spatial dimensions as x
        if mask.shape != x.shape:
            mask = F.interpolate(mask.float(), size=x.shape[2:], mode="nearest").bool()

        # Apply mask
        x = x * mask

        # Scale the activations
        # To maintain the expected value, scale by (mask.numel() / mask.sum())
        # Add a small epsilon to prevent division by zero
        eps = 1e-7
        x = x * (mask.numel() / (mask.sum() + eps))

        return x  # noqa: RET504


class KantoDexClassifierCustom(nn.Module):
    """
    KantoDexClassifierCustom is a custom neural network model for classifying Generation I Pokémon.

    It combines convolutional layers with residual connections, SE blocks,
    multi-head self-attention, positional encoding, and DropBlock regularization.
    """

    def __init__(  # noqa: PLR0913
        self,
        num_classes: int = 151,
        drop_prob: float = 0.4,
        attention_embed_dim: int = 512,  # Set to 512 to match convolutional output
        attention_num_heads: int = 8,
        dropblock_block_size: int = 7,
        max_len: int = 10000,
    ) -> None:
        """
        Initialize the KantoDexClassifierCustom.

        Args:
            num_classes (int): Number of output classes.
            drop_prob (float): Dropout rate for regularization.
            attention_embed_dim (int): Embedding dimension for attention.
            attention_num_heads (int): Number of attention heads.
            dropblock_block_size (int): Block size for DropBlock.
            max_len (int): Maximum sequence length for Positional Encoding.

        """
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(
                3,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),  # Output: (64, H/2, W/2)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # Output: (64, H/4, W/4)
        )

        # Convolutional Blocks
        self.layer1 = self._make_layer(64, 128, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(128, 256, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(256, 512, num_blocks=2, stride=2)

        # DropBlock after convolutional layers
        self.dropblock = DropBlock2D(block_size=dropblock_block_size, drop_prob=drop_prob)

        # Projection to match attention_embed_dim (already 512, so can skip if not reducing)
        # If attention_embed_dim != 512 and you want to project, apply projection
        if attention_embed_dim != 512:  # noqa: PLR2004
            self.projection = nn.Linear(512, attention_embed_dim)
        else:
            self.projection = None

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(
            embed_dim=attention_embed_dim,
            max_len=max_len,
        )

        # Attention
        self.attention = MultiHeadSelfAttention(
            embed_dim=attention_embed_dim,
            num_heads=attention_num_heads,
        )

        # Adaptive Pooling and Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=drop_prob)
        self.classifier = nn.Linear(attention_embed_dim, num_classes)

    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        """
        Create a layer with multiple ResidualConvBlocks.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            num_blocks (int): Number of residual blocks.
            stride (int): Stride for the first block.

        Returns:
            nn.Sequential: Sequential container of residual blocks.

        """
        layers = []
        layers.append(ResidualConvBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualConvBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, H, W)

        Returns:
            torch.Tensor: Output predictions of shape (batch_size, num_classes)

        """
        x = self.stem(x)  # Shape: (batch, 64, H/4, W/4)
        x = self.layer1(x)  # Shape: (batch, 128, H/4, W/4)
        x = self.layer2(x)  # Shape: (batch, 256, H/8, W/8)
        x = self.layer3(x)  # Shape: (batch, 512, H/16, W/16)

        # Apply DropBlock after convolutional layers
        x = self.dropblock(x)  # Shape: (batch, 512, H/16, W/16)

        # Prepare for Attention: Flatten spatial dimensions
        batch, channels, height, width = x.size()
        x = x.view(batch, channels, height * width).permute(
            0,
            2,
            1,
        )  # Shape: (batch, seq_length, channels)

        # Optional Projection
        if self.projection is not None:
            x = self.projection(x)  # Shape: (batch, seq_length, attention_embed_dim)

        # Positional Encoding
        x = self.positional_encoding(x)  # Shape: (batch, seq_length, embed_dim)

        # Attention
        x = self.attention(x)  # Shape: (batch, seq_length, embed_dim)

        # Global Average Pooling over sequence length
        x = x.mean(dim=1)  # Shape: (batch, embed_dim)

        # Final Classification Layers
        x = self.avgpool(x.view(batch, -1, 1, 1)).view(batch, -1)  # Shape: (batch, embed_dim)
        x = self.dropout(x)
        x = self.classifier(x)  # Shape: (batch, num_classes)

        return x  # noqa: RET504
