import argparse
import logging
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn, optim
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data.data_loader import collate_fn, load_data
from src.models.model import KantoDexClassifier
from src.utils.helpers import save_checkpoint
from src.utils.metrics import MetricsCalculator
from src.visualization.visualize_model import visualize_model_structure


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
    parser.add_argument(
        "--visualize_model",
        action="store_true",
        help="Visualize the model architecture.",
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
        drop_prob=config["model"].get("dropout", 0.1),
        custom_config=config["model"].get("custom_model_params", None),
    ).to(device)
    logging.info(
        "Initialized model: {}".format(
            config["model"].get("name", "efficientnet_b3"),
        ),
    )
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
    optimizer_name = config["training"].get("optimizer", "adamw").lower()
    if optimizer_name == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config["training"]["optimizer_params"].get("learning_rate", 1e-4),
            weight_decay=config["training"]["optimizer_params"].get("weight_decay", 0.01),
            eps=config["training"]["optimizer_params"].get("eps", 1e-8),
        )
        logging.info("Initialized AdamW optimizer.")
    elif optimizer_name == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config["training"]["optimizer_params"].get("learning_rate", 1e-4),
            weight_decay=config["training"]["optimizer_params"].get("weight_decay", 0.01),
            eps=config["training"]["optimizer_params"].get("eps", 1e-8),
        )
        logging.info("Initialized Adam optimizer.")
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config["training"]["optimizer_params"].get("learning_rate", 0.01),
            momentum=config["training"]["optimizer_params"].get("momentum", 0.9),
            weight_decay=config["training"]["optimizer_params"].get("weight_decay", 0.01),
        )
        logging.info("Initialized SGD optimizer.")
    elif optimizer_name == "rmsprop":
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=config["training"]["optimizer_params"].get("learning_rate", 0.01),
            alpha=config["training"]["optimizer_params"].get("alpha", 0.99),
            eps=config["training"]["optimizer_params"].get("eps", 1e-8),
            weight_decay=config["training"]["optimizer_params"].get("weight_decay", 0.01),
            momentum=config["training"]["optimizer_params"].get("momentum", 0.9),
        )
        logging.info("Initialized RMSprop optimizer.")
    else:
        msg = f"Unknown optimizer: {optimizer_name}"
        raise ValueError(msg)
    return optimizer


def initialize_scheduler(
    config: dict[str, Any],
    optimizer: optim.Optimizer,
) -> optim.lr_scheduler._LRScheduler | None:
    """
    Initialize the learning rate scheduler based on the configuration.

    Args:
        config (Dict[str, Any]): Configuration parameters.
        optimizer (optim.Optimizer): The optimizer.

    Returns:
        Optional[optim.lr_scheduler._LRScheduler]: Initialized scheduler or None.

    """
    scheduler = None
    scheduler_name = config["training"].get("scheduler", None)

    if scheduler_name == "step_lr":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config["training"].get("scheduler_params", {}).get("step_size", 30),
            gamma=config["training"].get("scheduler_params", {}).get("gamma", 0.1),
        )
        logging.info("Initialized StepLR scheduler.")
    if scheduler_name == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["training"].get("scheduler_params", {}).get("T_max", 10),
            eta_min=config["training"].get("scheduler_params", {}).get("eta_min", 1e-6),
        )
        logging.info("Initialized CosineAnnealingLR scheduler.")
    if scheduler_name == "cosine_annealing_warm_restarts":
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config["training"].get("scheduler_params", {}).get("T_0", 10),
            T_mult=config["training"].get("scheduler_params", {}).get("T_mult", 2),
            eta_min=config["training"].get("scheduler_params", {}).get("eta_min", 1e-6),
        )
        logging.info("Initialized CosineAnnealingWarmRestarts scheduler.")
    if scheduler_name == "reduce_lr_on_plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config["training"].get("scheduler_params", {}).get("mode", "min"),
            factor=config["training"].get("scheduler_params", {}).get("factor", 0.1),
            patience=config["training"].get("scheduler_params", {}).get("patience", 10),
            threshold=config["training"].get("scheduler_params", {}).get("threshold", 1e-4),
            threshold_mode=config["training"]
            .get("scheduler_params", {})
            .get("threshold_mode", "rel"),
            cooldown=config["training"].get("scheduler_params", {}).get("cooldown", 0),
            min_lr=config["training"].get("scheduler_params", {}).get("min_lr", 0),
            eps=config["training"].get("scheduler_params", {}).get("eps", 1e-8),
            verbose=config["training"].get("scheduler_params", {}).get("verbose", False),
        )
        logging.info("Initialized ReduceLROnPlateau scheduler.")
    if scheduler:
        logging.info(f"Initialized scheduler: {scheduler_name}")
    else:
        logging.info("No scheduler initialized.")
    return scheduler


def resume_training(  # noqa: PLR0913
    config: dict[str, Any],
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler | None,
    scaler: GradScaler,
    device: torch.device,
) -> tuple[int, float]:
    """
    Resume training from the latest checkpoint if available.

    Args:
        config (Dict[str, Any]): Configuration parameters.
        model (nn.Module): The model.
        optimizer (optim.Optimizer): The optimizer.
        scheduler (Optional[_LRScheduler]): The scheduler.
        scaler (GradScaler): The gradient scaler.
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
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
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


def train_epoch(  # noqa:  PLR0913
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    metrics_calculator: Any,
    use_cutmix: bool = False,
    use_mixup: bool = False,
    alpha: float = 1.0,
) -> tuple[float, float]:
    """
    Train the model for one epoch with optional CutMix and MixUp augmentations.

    Args:
        model (nn.Module): The model.
        dataloader (DataLoader): Training data loader.
        criterion (nn.Module): Loss function.
        optimizer (optim.Optimizer): Optimizer.
        device (torch.device): Device to train on.
        scaler (GradScaler): Gradient scaler for mixed precision.
        metrics_calculator (Any): Metrics calculator.
        use_cutmix (bool, optional): Flag to use CutMix augmentation.
        use_mixup (bool, optional): Flag to use MixUp augmentation.
        alpha (float, optional): Alpha parameter for CutMix and MixUp.

    Returns:
        Tuple[float, float]: (Average loss, Average accuracy) for the epoch.

    """
    model.train()
    running_loss = 0.0

    for batch in tqdm(dataloader, desc="Training", unit="batch"):
        images, labels = batch

        if use_cutmix or use_mixup:
            images, labels = collate_fn(images, labels, use_cutmix, use_mixup, alpha)

        images = images.to(device)
        labels = labels.to(device)

        # **Convert labels to Long dtype**
        labels = labels.long()

        optimizer.zero_grad()

        with autocast(device.type, enabled=device.type == "cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        metrics_calculator.update(outputs, labels)

    avg_loss = running_loss / len(dataloader)
    metrics = metrics_calculator.compute()
    return avg_loss, metrics["accuracy"]


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    metrics_calculator: Any,
) -> float:
    """
    Validate the model.

    Args:
        model (nn.Module): The model.
        dataloader (DataLoader): Validation data loader.
        device (torch.device): Device to validate on.
        metrics_calculator (Any): Metrics calculator.

    Returns:
        float: Validation accuracy in percentage.

    """
    model.eval()
    metrics_calculator.reset()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", unit="batch"):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            metrics_calculator.update(outputs, labels)
    metrics = metrics_calculator.compute()
    return metrics["accuracy"]


def main() -> None:  # noqa: PLR0915
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
        use_cutmix=config["augmentation"].get("use_cutmix", False),
        use_mixup=config["augmentation"].get("use_mixup", False),
        alpha=config["augmentation"].get("alpha", 1.0),
    )
    num_classes = len(label_to_idx)
    logging.info(f"Number of classes: {num_classes}")

    # Initialize model, optimizer, scheduler
    model = initialize_model(config, num_classes, device)
    optimizer = initialize_optimizer(config, model)
    scheduler = initialize_scheduler(config, optimizer)

    # Loss function with Label Smoothing
    label_smoothing = config["training"].get("label_smoothing", 0.1)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    metrics_calculator = MetricsCalculator(num_classes=num_classes)

    # TensorBoard writer
    writer = SummaryWriter(log_dir="runs")

    # Mixed Precision Scaler
    scaler = GradScaler("cuda", enabled=device.type == "cuda")

    # Early Stopping parameters
    early_stopping_patience = config["training"].get("early_stopping_patience", 10)
    epochs_no_improve = 0
    best_accuracy = 0.0

    # Resume training if flag is set
    start_epoch = 0
    if args.resume:
        start_epoch, best_accuracy = resume_training(
            config,
            model,
            optimizer,
            scheduler,
            scaler,
            device,
        )

    # Visualize model structure if flag is set
    if args.visualize_model:
        visualize_model_structure(model)

    num_epochs = config["training"]["epochs"]
    for epoch in range(start_epoch, num_epochs):
        logging.info(f"Starting epoch {epoch + 1}/{num_epochs}")

        # Train
        train_loss, train_accuracy = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            metrics_calculator,
            use_cutmix=config["training"].get("use_cutmix", False),
            use_mixup=config["training"].get("use_mixup", False),
            alpha=config["training"].get("alpha", 1.0),
        )
        logging.info(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%",  # noqa: E501
        )
        writer.add_scalar("Training Loss", train_loss, epoch + 1)
        writer.add_scalar("Training Accuracy", train_accuracy, epoch + 1)

        precision = metrics_calculator.precision.compute().item()
        recall = metrics_calculator.recall.compute().item()
        f1_score = metrics_calculator.f1.compute().item()
        writer.add_scalar("Precision", precision, epoch + 1)
        writer.add_scalar("Recall", recall, epoch + 1)
        writer.add_scalar("F1 Score", f1_score, epoch + 1)

        # Scheduler step
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(train_loss)
            else:
                scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            logging.info(f"Learning Rate: {current_lr}")
            writer.add_scalar("Learning Rate", current_lr, epoch + 1)

        # Validate
        val_accuracy = validate(model, val_loader, device, metrics_calculator)
        logging.info(f"Validation Accuracy: {val_accuracy:.2f}%")
        writer.add_scalar("Validation Accuracy", val_accuracy, epoch + 1)

        # Check for best model
        is_best = val_accuracy > best_accuracy
        if is_best:
            best_accuracy = val_accuracy
            epochs_no_improve = 0
            logging.info(f"New best accuracy: {best_accuracy:.2f}%")
        else:
            epochs_no_improve += 1
            logging.info(f"No improvement in validation accuracy for {epochs_no_improve} epochs.")

        # Save checkpoint
        checkpoint_dir = Path(config["training"]["checkpoint_path"])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            checkpoint_dir=str(checkpoint_dir),
            scheduler=scheduler,
            scaler=scaler,
            is_best=is_best,
            best_accuracy=best_accuracy,
        )
        writer.add_scalar("Best Accuracy", best_accuracy, epoch + 1)

        # Early Stopping
        if epochs_no_improve >= early_stopping_patience:
            logging.info("Early stopping triggered.")
            break

    logging.info("Training complete.")
    writer.close()


if __name__ == "__main__":
    main()
