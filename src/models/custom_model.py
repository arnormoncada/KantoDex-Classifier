import torch
import torch.nn.functional as F
from torch import nn


class LearnablePositionalEncoding(nn.Module):
    """Learnable Positional Encoding module adds positional information to the input embeddings."""

    def __init__(self, embed_dim: int, max_len: int = 10000) -> None:
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, embed_dim)
        self.max_len = max_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add learnable positional encoding to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_length, embed_dim)

        Returns:
            torch.Tensor: Tensor with positional encoding added.

        """
        batch_size, seq_length, embed_dim = x.size()
        position_ids = (
            torch.arange(seq_length, dtype=torch.long, device=x.device)
            .unsqueeze(0)
            .expand(batch_size, seq_length)
        )
        pos_emb = self.pos_embedding(position_ids)
        x = x + pos_emb
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
        self.attention = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        # x shape: (batch, seq_length, embed_dim)
        attn_output, _ = self.attention(x, x, x)
        x = self.norm(x + self.dropout(attn_output))
        return x  # noqa: RET504


class TransformerEncoderLayer(nn.Module):
    """Transformer Encoder Layer with MHSA and Feed-Forward Network."""

    def __init__(
        self, embed_dim: int, num_heads: int, ff_hidden_dim: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.mhsa = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        x = self.mhsa(x)
        ffn_output = self.ffn(x)
        x = self.norm(x + ffn_output)
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

    It uses a custom architecture with a convolutional backbone, attention mechanism, SE blocks,
    transformers, multi-head self-attention, positional encoding, and DropBlock regularization.
    """

    def __init__(  # noqa: PLR0913
        self,
        num_classes: int = 151,
        drop_prob: float = 0.4,
        attention_embed_dim: int = 512,  # Set to 512 to match convolutional output
        attention_num_heads: int = 8,
        transformer_layers: int = 4,
        ff_hidden_dim: int = 2048,
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
            transformer_layers (int): Number of transformer encoder layers.
            ff_hidden_dim (int): Hidden dimension size for feed-forward networks in transformer.
            dropblock_block_size (int): Block size for DropBlock.
            max_len (int): Maximum sequence length for Positional Encoding.

        """
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(
                3,
                128,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),  # Output: (128, H/2, W/2)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # Output: (128, H/4, W/4)
        )

        # Convolutional Blocks
        self.layer1 = self._make_layer(128, 256, num_blocks=3, stride=1)  # Increased channels
        self.layer2 = self._make_layer(256, 512, num_blocks=4, stride=2)
        self.layer3 = self._make_layer(
            512,
            1024,
            num_blocks=6,
            stride=2,
        )  # Further increased channels

        # DropBlock after convolutional layers
        self.dropblock = DropBlock2D(block_size=dropblock_block_size, drop_prob=drop_prob)

        # Classification Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, attention_embed_dim))

        # Projection to match attention_embed_dim
        self.projection = nn.Conv2d(
            1024,
            attention_embed_dim,
            kernel_size=1,
        )  # Changed to Conv2d for spatial projection

        # Positional Encoding
        self.positional_encoding = LearnablePositionalEncoding(
            embed_dim=attention_embed_dim,
            max_len=max_len,
        )

        # Transformer Encoder
        self.transformer_encoder_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    embed_dim=attention_embed_dim,
                    num_heads=attention_num_heads,
                    ff_hidden_dim=ff_hidden_dim,
                    dropout=drop_prob,
                )
                for _ in range(transformer_layers)
            ],
        )

        # Dropout
        self.dropout = nn.Dropout(p=drop_prob)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(attention_embed_dim, ff_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_prob),
            nn.Linear(ff_hidden_dim, num_classes),
        )

        # Initialize CLS token
        nn.init.trunc_normal_(self.cls_token, std=0.02)

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
        batch = x.size(0)
        x = self.stem(x)  # Shape: (batch, 128, H/4, W/4)
        x = self.layer1(x)  # Shape: (batch, 256, H/4, W/4)
        x = self.layer2(x)  # Shape: (batch, 512, H/8, W/8)
        x = self.layer3(x)  # Shape: (batch, 1024, H/16, W/16)

        # Apply DropBlock after convolutional layers
        x = self.dropblock(x)  # Shape: (batch, 1024, H/16, W/16)

        # Projection to attention_embed_dim
        x = self.projection(x)  # Shape: (batch, attention_embed_dim, H/16, W/16)

        # Flatten spatial dimensions
        batch, embed_dim, height, width = x.size()
        seq_length = height * width
        x = x.view(batch, embed_dim, seq_length).permute(
            0,
            2,
            1,
        )  # Shape: (batch, seq_length, embed_dim)

        # Concatenate CLS token
        cls_tokens = self.cls_token.expand(batch, -1, -1)  # Shape: (batch, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # Shape: (batch, seq_length + 1, embed_dim)

        # Positional Encoding
        x = self.positional_encoding(x)  # Shape: (batch, seq_length + 1, embed_dim)

        # Transformer Encoder
        for layer in self.transformer_encoder_layers:
            x = layer(x)  # Shape: (batch, seq_length + 1, embed_dim)

        # Extract CLS token for classification
        cls_output = x[:, 0, :]  # Shape: (batch, embed_dim)

        # Final Classification Layers
        x = self.dropout(cls_output)
        x = self.classifier(x)  # Shape: (batch, num_classes)

        return x  # noqa: RET504
