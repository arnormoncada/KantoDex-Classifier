import argparse
import logging
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data.data_loader import DataLoader, load_data
from src.models.model import KantoDexClassifier
from src.utils.helpers import save_checkpoint


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.

    """
    parser = argparse.ArgumentParser(description="Train the KantoDex Classifier model.")
    parser.add_argument(
        "--config",
        type=str,
        default="src/config/config.yaml",
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Flag to resume training from the latest checkpoint.",
    )
    return parser.parse_args()


def setup_logging(log_dir: str = "logs") -> None:
    """
    Set up logging to file and console.

    Args:
        log_dir (str): Directory to save log files.

    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=Path(log_dir) / "training.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    # Also log to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)


def load_config(config_path: str) -> dict[str, Any]:
    """
    Load the YAML configuration file.

    Args:
        config_path (str): Path to the YAML config file.

    Returns:
        Dict[str, Any]: Configuration parameters.

    """
    with open(config_path) as f:
        return yaml.safe_load(f)


def initialize_model(config: dict[str, Any], num_classes: int, device: torch.device) -> nn.Module:
    """
    Initialize the model based on the configuration.

    Args:
        config (Dict[str, Any]): Configuration parameters.
        num_classes (int): Number of output classes.
        device (torch.device): Device to load the model on.

    Returns:
        nn.Module: Initialized model.

    """
    model = KantoDexClassifier(
        model_name=config["model"].get("name", "efficientnet_b3"),
        num_classes=num_classes,
        pretrained=config["model"].get("pretrained", True),
        dropout=config["model"].get("dropout", 0.4),
    ).to(device)
    logging.info("Initialized model: {}".format(config["model"].get("name", "efficientnet_b3")))
    return model


def initialize_optimizer(config: dict[str, Any], model: nn.Module) -> optim.Optimizer:
    """
    Initialize the optimizer based on the configuration.

    Args:
        config (Dict[str, Any]): Configuration parameters.
        model (nn.Module): The model to optimize.

    Returns:
        optim.Optimizer: Initialized optimizer.

    """
    optimizer_name = config["training"].get("optimizer", "adam").lower()
    lr = config["training"]["learning_rate"]
    if optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        error_msg = f"Unsupported optimizer: {optimizer_name}"
        raise ValueError(error_msg)
    logging.info(f"Initialized optimizer: {optimizer_name} with learning rate {lr}")
    return optimizer


def initialize_scheduler(
    config: dict[str, Any],
    optimizer: optim.Optimizer,
    num_epochs: int,
) -> optim.lr_scheduler._LRScheduler | None:
    """
    Initialize the learning rate scheduler based on the configuration.

    Args:
        config (Dict[str, Any]): Configuration parameters.
        optimizer (optim.Optimizer): The optimizer.
        num_epochs (int): Total number of training epochs.

    Returns:
        Optional[optim.lr_scheduler._LRScheduler]: Initialized scheduler or None.

    """
    scheduler = None
    scheduler_name = config["training"].get("scheduler", None)
    if scheduler_name == "step_lr":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config["training"]["step_size"],
            gamma=config["training"]["gamma"],
        )
        logging.info("Initialized StepLR scheduler.")
    elif scheduler_name == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        logging.info("Initialized CosineAnnealingLR scheduler.")
    return scheduler


def resume_training(
    config: dict[str, Any],
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler | None,
    device: torch.device,
) -> tuple[int, float]:
    """
    Resume training from the latest checkpoint if available.

    Args:
        config (Dict[str, Any]): Configuration parameters.
        model (nn.Module): The model.
        optimizer (optim.Optimizer): The optimizer.
        scheduler (Optional[_LRScheduler]): The scheduler.
        device (torch.device): Device to load the checkpoint on.

    Returns:
        Tuple[int, float]: (start_epoch, best_accuracy)

    """
    checkpoint_dir = Path(config["training"]["checkpoint_path"])
    checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*.pth"))
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
        logging.info(f"Resuming from checkpoint: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_accuracy = checkpoint.get("best_accuracy", 0.0)
        logging.info(
            f"Resumed training from epoch {start_epoch} with best accuracy {best_accuracy:.2f}%",
        )
        return start_epoch, best_accuracy
    logging.warning("No checkpoint found to resume.")
    return 0, 0.0


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The model.
        dataloader (DataLoader): Training data loader.
        criterion (nn.Module): Loss function.
        optimizer (optim.Optimizer): Optimizer.
        device (torch.device): Device to train on.

    Returns:
        float: Average loss for the epoch.

    """
    model.train()
    running_loss = 0.0
    for batch in tqdm(dataloader, desc="Training", unit="batch"):
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    avg_loss = running_loss / len(dataloader)
    return avg_loss  # noqa: RET504


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """
    Validate the model.

    Args:
        model (nn.Module): The model.
        dataloader (DataLoader): Validation data loader.
        device (torch.device): Device to validate on.

    Returns:
        float: Validation accuracy in percentage.

    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", unit="batch"):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy  # noqa: RET504


def main() -> None:
    """Orchestrate the training process."""
    args = parse_args()
    setup_logging()
    config = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load data
    train_loader, val_loader, label_to_idx = load_data(
        processed_path=config["data"]["processed_path"],
        test_size=config["data"]["test_size"],
        batch_size=config["training"]["batch_size"],
        img_size=tuple(config["data"]["img_size"]),
        num_workers=config["data"]["num_workers"],
    )

    num_classes = len(label_to_idx)
    logging.info(f"Number of classes: {num_classes}")

    # Initialize model, optimizer, scheduler
    model = initialize_model(config, num_classes, device)
    optimizer = initialize_optimizer(config, model)
    scheduler = initialize_scheduler(config, optimizer, config["training"]["epochs"])

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # TensorBoard
    writer = SummaryWriter(log_dir="runs")

    # Resume training if flag is set
    start_epoch, best_accuracy = 0, 0.0
    if args.resume:
        start_epoch, best_accuracy = resume_training(config, model, optimizer, scheduler, device)

    num_epochs = config["training"]["epochs"]
    for epoch in range(start_epoch, num_epochs):
        logging.info(f"Starting epoch {epoch + 1}/{num_epochs}")

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        logging.info(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}")
        writer.add_scalar("Training Loss", train_loss, epoch + 1)

        # Scheduler step
        if scheduler:
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            logging.info(f"Learning Rate: {current_lr}")
            writer.add_scalar("Learning Rate", current_lr, epoch + 1)

        # Validate
        val_accuracy = validate(model, val_loader, device)
        logging.info(f"Validation Accuracy: {val_accuracy:.2f}%")
        writer.add_scalar("Validation Accuracy", val_accuracy, epoch + 1)

        # Check for best model
        is_best = val_accuracy > best_accuracy
        if is_best:
            best_accuracy = val_accuracy
            logging.info(f"New best accuracy: {best_accuracy:.2f}%")

        # Save checkpoint
        checkpoint_dir = Path(config["training"]["checkpoint_path"])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            checkpoint_dir=str(checkpoint_dir),
            scheduler=scheduler,
            is_best=is_best,
            best_accuracy=best_accuracy,
        )
        writer.add_scalar("Best Accuracy", best_accuracy, epoch + 1)

    logging.info("Training complete.")
    writer.close()


if __name__ == "__main__":
    main()
