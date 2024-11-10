import torch
from torchmetrics import Accuracy, F1Score, Precision, Recall


class MetricsCalculator:
    """A class to calculate and store various metrics for model evaluation."""

    def __init__(self, num_classes: int) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.correct = [0] * self.num_classes
        self.total = [0] * self.num_classes

        # Initialize metrics with the 'task' parameter
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)
        self.precision = Precision(task="multiclass", num_classes=num_classes, average="macro").to(
            device,
        )
        self.recall = Recall(task="multiclass", num_classes=num_classes, average="macro").to(device)
        self.f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro").to(device)

    def reset(self):
        """Reset all the metrics to their initial state."""
        self.correct = [0] * self.num_classes
        self.total = [0] * self.num_classes
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
        for label, pred in zip(labels, preds, strict=False):
            if label == pred:
                self.correct[label] += 1
            self.total[label] += 1

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
        overall_accuracy = self.accuracy.compute().item() * 100
        precision = self.precision.compute().item() * 100
        recall = self.recall.compute().item() * 100
        f1_score = self.f1.compute().item() * 100

        per_class_accuracy = [
            (c / t * 100) if t > 0 else 0.0 for c, t in zip(self.correct, self.total, strict=False)
        ]

        return {
            "accuracy": overall_accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1_score,
            "per_class_accuracy": per_class_accuracy,
        }
