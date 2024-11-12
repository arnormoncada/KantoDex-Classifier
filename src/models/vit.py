import torch
from torch import nn


class PatchEmbedding(nn.Module):
    """Splits image into patches and embeds them."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 14,
        in_channels: int = 3,
        embed_dim: int = 1280,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        # Linear projection of flattened patches
        self.flatten = nn.Flatten(2)  # (B, C, N)
        # Learnable class token
        self.cls_token = nn.Parameter(torch.randn(size=(1, 1, embed_dim)), requires_grad=True)
        self.position_embeddings = nn.Parameter(
            torch.randn(size=(1, self.num_patches + 1, embed_dim)),
            requires_grad=True,
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        # Copy of cls token for each element in the batch
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)  # (B, 1, C)

        # Patch Embedding
        x = self.proj(x)  # (B, C, H, W)
        x = self.flatten(x)  # (B, C, N)
        x = x.permute(0, 2, 1)  # (B, N, C)

        # Prepend class token to the input
        x = torch.cat((cls_tokens, x), dim=1)  # (B, 1+N, C)

        # Add position embeddings
        x = x + self.position_embeddings  # (B, 1+N, C)

        # Apply dropout
        x = self.dropout(x)
        return x  # noqa: RET504


class MLP(nn.Module):
    """Multi-Layer Perceptron for the Transformer Encoder Block."""

    def __init__(self, embed_dim: int, expansion: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim * expansion)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(embed_dim * expansion, embed_dim)
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.fc1,
            self.gelu,
            self.dropout1,
            self.fc2,
            self.dropout2,
        )

    def forward(self, x):  # noqa: D102
        x = self.net(x)
        return x  # noqa: RET504


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention Layer."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            msg = "Embedding dimension must be divisible by number of heads"
            raise ValueError(msg)
        self.head_dim = embed_dim // num_heads

        self.W_q = nn.Linear(embed_dim, embed_dim)  # Query
        self.W_k = nn.Linear(embed_dim, embed_dim)  # Key
        self.W_v = nn.Linear(embed_dim, embed_dim)  # Value
        self.W_o = nn.Linear(embed_dim, embed_dim)  # Output
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Split the last dimension into (num_heads, head_dim)."""
        batch_size, seq_length, _ = x.size()
        x = x.view(batch_size, seq_length, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_length, head_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:  # noqa: D102
        batch_size, seq_length, _ = x.size()

        # Linear projections
        queries = self.W_q(x)
        keys = self.W_k(x)
        values = self.W_v(x)

        # Split heads
        queries = self.split_heads(queries)
        keys = self.split_heads(keys)
        values = self.split_heads(values)

        # Scaled dot-product attention
        scaling = self.head_dim**0.5
        energy = torch.matmul(queries, keys.transpose(-2, -1)) / scaling

        # Apply mask (optional)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-inf"))

        # Apply the Softmax
        attention = torch.softmax(energy, dim=-1)
        attention = self.dropout(attention)

        # Attention output
        x = torch.matmul(attention, values)

        # Concatenate heads
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, seq_length, self.embed_dim)

        # Final linear projection
        x = self.W_o(x)

        return x  # noqa: RET504


class TransformerEncoderLayer(nn.Module):
    """Single Transformer Encoder Layer."""

    def __init__(self, embed_dim, num_heads, expansion, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.MHA = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, expansion, dropout)

    def forward(self, x):  # noqa: D102
        norm1 = self.norm1(x)
        # Skip connection 1
        x = x + self.MHA(norm1)
        x = self.dropout(x)

        norm2 = self.norm2(x)
        # Skip connection 2
        x = x + self.mlp(norm2)
        x = self.dropout(x)

        return x  # noqa: RET504


class TransformerEncoder(nn.Module):  # noqa: D101
    def __init__(self, embed_dim, num_heads, expansion, dropout, num_encoders):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(embed_dim, num_heads, expansion, dropout)
                for _ in range(num_encoders)
            ],
        )

    def forward(self, x):  # noqa: D102
        for layer in self.layers:
            x = layer(x)
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer for Image Classification."""

    def __init__(  # noqa: PLR0913
        self,
        img_size: int = 224,
        patch_size: int = 14,
        in_channels: int = 3,
        num_classes: int = 151,
        embed_dim: int = 1280,
        num_encoders: int = 32,
        num_heads: int = 16,
        expansion: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # Patch Embedding
        self.patch_embeddings = PatchEmbedding(
            img_size,
            patch_size,
            in_channels,
            embed_dim,
            dropout,
        )
        # Transformer Encoder Layers
        self.transformer = TransformerEncoder(
            embed_dim,
            num_heads,
            expansion,
            dropout,
            num_encoders,
        )

        # Classification Head
        self.classification_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        x = self.patch_embeddings(x)
        x = self.transformer(x)
        x = self.classification_head(x[:, 0, :])  # Take the cls token
        return x  # noqa: RET504

    def _init_weights(self) -> None:
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.MultiheadAttention):
                nn.init.xavier_uniform_(m.in_proj_weight)
                nn.init.xavier_uniform_(m.out_proj.weight)
                if m.in_proj_bias is not None:
                    nn.init.constant_(m.in_proj_bias, 0)
                    nn.init.constant_(m.out_proj.bias, 0)


# Example usage
if __name__ == "__main__":
    # Create a random tensor with shape (batch_size, channels, height, width)
    batch_size = 8
    channels = 3
    height = 224
    width = 224
    num_classes = 151

    model = VisionTransformer()

    inputs = torch.randn(batch_size, channels, height, width)
    outputs = model(inputs)
    print(f"Output shape: {outputs.shape}")  # Expected: (8, 151)

    # Print model summary
    print(model)
