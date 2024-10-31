from torch import Tensor, nn
from torchvision import models


class KantoDexClassifier(nn.Module):
    """
    KantoDexClassifier is a neural network model for classifying Generation I Pokémon.

    Attributes:
        backbone (nn.Module): The backbone model (e.g., EfficientNet B3 or ResNet50).

    """

    def __init__(
        self,
        model_name: str = "efficientnet_b3",
        num_classes: int = 151,
        pretrained: bool = True,
        dropout: float = 0.4,
    ) -> None:
        """
        Initialize the KantoDexClassifier with the specified backbone model.

        Args:
            model_name (str): Name of the backbone model to use ('efficientnet_b3' or 'resnet50').
            num_classes (int): Number of output classes.
            pretrained (bool): Whether to use pretrained weights.
            dropout (float): Dropout rate for regularization.

        """
        super().__init__()
        self.model_name = model_name
        if model_name == "efficientnet_b3":
            weights = "EfficientNet_B3_Weights.DEFAULT" if pretrained else None
            self.backbone = models.efficientnet_b3(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=dropout, inplace=True),
                nn.Linear(in_features, num_classes),
            )
        elif model_name == "resnet50":
            weights = "ResNet50_Weights.DEFAULT" if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes),
            )
        else:
            error_msg = f"Unsupported model_name: {model_name}"
            raise ValueError(error_msg)

    def forward(self, x: Tensor) -> Tensor:
        """
        Perform a forward pass of the model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output predictions.

        """
        return self.backbone(x)
