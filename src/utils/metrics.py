import torch
from torchmetrics import Accuracy, F1Score, Precision, Recall


class MetricsCalculator:
    """A class to calculate and store various metrics for model evaluation."""

    def __init__(self, num_classes: int) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize metrics with the 'task' parameter
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)
        self.precision = Precision(task="multiclass", num_classes=num_classes, average="macro").to(
            device,
        )
        self.recall = Recall(task="multiclass", num_classes=num_classes, average="macro").to(device)
        self.f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro").to(device)

    def reset(self):
        """Reset all the metrics to their initial state."""
        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1.reset()

    def update(self, outputs: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Update the metrics with the given outputs and labels.

        Args:
            outputs (torch.Tensor): The model outputs.
            labels (torch.Tensor): The ground truth labels.

        """
        preds = torch.argmax(outputs, dim=1)
        self.accuracy.update(preds, labels)
        self.precision.update(preds, labels)
        self.recall.update(preds, labels)
        self.f1.update(preds, labels)

    def compute(self) -> dict[str, float]:
        """
        Compute and return the metrics.

        Returns:
            dict[str, float]: A dictionary containing the computed metrics.

        """
        return {
            "accuracy": self.accuracy.compute().item() * 100,
            "precision": self.precision.compute().item() * 100,
            "recall": self.recall.compute().item() * 100,
            "f1_score": self.f1.compute().item() * 100,
        }
