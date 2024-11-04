import torch
from torch import nn


class PatchEmbedding(nn.Module):
    """Splits image into patches and embeds them."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.flatten = nn.Flatten(2)  # (B, C, N)
        self.transpose = lambda x: x.transpose(1, 2)  # (B, N, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = self.flatten(x)  # (B, embed_dim, N)
        x = self.transpose(x)  # (B, N, embed_dim)
        return x  # noqa: RET504


class PositionalEncoding(nn.Module):
    """Adds positional information to patch embeddings."""

    def __init__(self, num_patches: int, embed_dim: int) -> None:
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return x + self.pos_embed[:, : x.size(1), :]


class ClassToken(nn.Module):
    """Prepends a class token to the patch embeddings."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)  # (B, 1, C)
        return torch.cat((cls_tokens, x), dim=1)  # (B, 1+N, C)


class TransformerEncoderBlock(nn.Module):
    """Single Transformer Encoder Block."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mhsa = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        # Self-Attention
        x_norm = self.norm1(x)
        # nn.MultiheadAttention expects (seq_len, batch, embed_dim)
        x_attn, _ = self.mhsa(
            x_norm.transpose(0, 1),
            x_norm.transpose(0, 1),
            x_norm.transpose(0, 1),
        )
        x_attn = x_attn.transpose(0, 1)
        x = x + self.dropout1(x_attn)

        # Feed-Forward Network
        x_norm = self.norm2(x)
        x_ffn = self.ffn(x_norm)
        x = x + x_ffn
        return x  # noqa: RET504


class VisionTransformer(nn.Module):
    """Vision Transformer for Image Classification."""

    def __init__(  # noqa: PLR0913
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 151,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.class_token = ClassToken(embed_dim)
        self.positional_encoding = PositionalEncoding(
            num_patches=(img_size // patch_size) ** 2,
            embed_dim=embed_dim,
        )

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(depth)
            ],
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        x = self.patch_embed(x)  # (B, N, C)
        x = self.class_token(x)  # (B, 1+N, C)
        x = self.positional_encoding(x)  # (B, 1+N, C)

        for block in self.transformer_blocks:
            x = block(x)  # (B, 1+N, C)

        x = self.norm(x)  # (B, 1+N, C)
        cls_token_final = x[:, 0]  # (B, C)
        x = self.head(cls_token_final)  # (B, num_classes)
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

    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=num_classes,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.1,
    )

    inputs = torch.randn(batch_size, channels, height, width)
    outputs = model(inputs)
    print(f"Output shape: {outputs.shape}")  # Expected: (8, 151)
