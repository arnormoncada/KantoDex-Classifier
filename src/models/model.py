from torch import Tensor, nn
from torchvision import models


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
    ) -> None:
        """
        Initialize the KantoDexClassifier with the specified backbone model.

        Args:
            model_name (str): Name of the backbone model to use ('efficientnet_b3', 'resnet50').
            num_classes (int): Number of output classes.
            pretrained (bool): Whether to use pretrained weights.
            drop_prob (float): Dropout rate for regularization.

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
        else:
            error_msg = f"Unsupported model_name: {model_name}"
            raise ValueError(error_msg)

        # Optional: Initialize weights for the new classifier
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
