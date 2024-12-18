import torch
from torchmetrics import (
    AUROC,
    Accuracy,
    ConfusionMatrix,
    F1Score,
    Precision,
    Recall,
)


class MetricsCalculator:
    """A class to calculate and store various metrics for model evaluation."""

    def __init__(self, num_classes: int) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.correct = {}
        self.total = {}

        # Initialize metrics with the 'task' parameter
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)
        self.precision = Precision(
            task="multiclass",
            num_classes=num_classes,
            average="macro",
        ).to(device)
        self.recall = Recall(
            task="multiclass",
            num_classes=num_classes,
            average="macro",
        ).to(device)
        self.f1 = F1Score(
            task="multiclass",
            num_classes=num_classes,
            average="macro",
        ).to(device)
        self.confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes).to(
            device,
        )
        self.auroc = AUROC(task="multiclass", num_classes=num_classes, average="macro").to(device)

    def reset(self):
        """Reset all the metrics to their initial state."""
        self.correct = {}
        self.total = {}
        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1.reset()
        self.confusion_matrix.reset()
        self.auroc.reset()

    def update(self, outputs: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Update the metrics with the given outputs and labels.

        Args:
            outputs (torch.Tensor): The model outputs.
            labels (torch.Tensor): The ground truth labels.

        """
        preds = torch.argmax(outputs, dim=1)
        probs = torch.softmax(outputs, dim=1)  # For AUROC

        # Ensure labels and preds are on the same device
        labels = labels.to(preds.device).long()

        # Update correct and total counts per class
        for label, pred in zip(labels, preds, strict=False):
            label_item = label.item()
            pred_item = pred.item()
            if label_item == pred_item:
                self.correct[label_item] = self.correct.get(label_item, 0) + 1
            self.total[label_item] = self.total.get(label_item, 0) + 1

        # Update torchmetrics metrics
        self.accuracy.update(preds, labels)
        self.precision.update(preds, labels)
        self.recall.update(preds, labels)
        self.f1.update(preds, labels)
        self.confusion_matrix.update(preds, labels)
        self.auroc.update(probs, labels)

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
        auroc_score = self.auroc.compute().item() * 100

        per_class_accuracy = {
            class_id: (self.correct.get(class_id, 0) / self.total.get(class_id, 1)) * 100
            for class_id in range(self.num_classes)  # Fixed here
        }

        worst_performing_classes = sorted(
            per_class_accuracy.items(),
            key=lambda x: x[1],
        )

        confusion_matrix = self.confusion_matrix.compute().cpu().numpy()

        return {
            "accuracy": overall_accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1_score,
            "auroc": auroc_score,
            "per_class_accuracy": per_class_accuracy,
            "worst_performing_classes": worst_performing_classes,
            "confusion_matrix": confusion_matrix,
        }
