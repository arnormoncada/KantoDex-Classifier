from torch import Tensor, nn
from torchvision import models

from src.models.custom_model import KantoDexClassifierCustom
from src.models.vit import VisionTransformer


class KantoDexClassifier(nn.Module):
    """
    KantoDexClassifier is a neural network model for classifying Generation I Pokémon.

    Attributes:
        backbone (nn.Module): The backbone model (e.g., EfficientNet B3, ResNet50).

    """

    def __init__(
        self,
        model_name: str = "efficientnet_b3",
        num_classes: int = 151,
        pretrained: bool = True,
        drop_prob: float = 0.4,
        custom_config: dict | None = None,
    ) -> None:
        """
        Initialize the KantoDexClassifier with the specified backbone model.

        Args:
            model_name (str): Name of the backbone model to use ('efficientnet_b3', 'resnet50').
            num_classes (int): Number of output classes.
            pretrained (bool): Whether to use pretrained weights.
            drop_prob (float): Dropout rate for regularization.
            custom_config (dict): Configuration for the custom model.

        """
        super().__init__()
        self.model_name = model_name

        if model_name == "efficientnet_b3":
            weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b3(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=drop_prob, inplace=True),
                nn.Linear(in_features, num_classes),
            )

        elif model_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=drop_prob),
                nn.Linear(in_features, num_classes),
            )
        elif model_name == "custom":
            self.backbone = KantoDexClassifierCustom(
                num_classes=num_classes,
                drop_prob=custom_config.get("drop_prob", drop_prob),
                attention_embed_dim=custom_config.get("attention_embed_dim", 512),
                attention_num_heads=custom_config.get("attention_num_heads", 8),
                dropblock_block_size=custom_config.get("dropblock_block_size", 7),
                max_len=custom_config.get("max_len", 10000),
            )
        elif model_name == "vit":
            self.backbone = VisionTransformer(
                img_size=224,
                patch_size=16,
                in_channels=3,
                num_classes=151,
                embed_dim=768,
                depth=12,
                num_heads=12,
                mlp_ratio=4.0,
                dropout=0.1,
            )
        else:
            msg = f"Invalid model name: {model_name}"
            raise ValueError(msg)
        # Optional: Initialize weights for the new classifier
        if model_name != "custom":
            self._initialize_weights_for_custom()
        else:
            self._initialize_weights()

    def forward(self, x: Tensor) -> Tensor:
        """
        Perform a forward pass of the model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output predictions.

        """
        return self.backbone(x)

    def _initialize_weights(self):
        """Initialize weights of the classifier layers if needed."""
        for m in self.backbone.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _initialize_weights_for_custom(self):
        """Initialize weights of the custom model layers if needed."""
        for m in self.backbone.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d | nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.MultiheadAttention):
                nn.init.xavier_uniform_(m.in_proj_weight)
                nn.init.xavier_uniform_(m.out_proj.weight)
                if m.in_proj_bias is not None:
                    nn.init.constant_(m.in_proj_bias, 0)
                    nn.init.constant_(m.out_proj.bias, 0)
