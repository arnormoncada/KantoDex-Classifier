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
from tqdm import tqdm

from src.data.data_loader import collate_fn, load_data
from src.models.model import KantoDexClassifier
from src.utils.helpers import save_checkpoint
from src.utils.metrics import MetricsCalculator
from src.visualization.tensorboard_logger import TensorBoardLogger
from src.visualization.visualize_model import visualize_model_structure


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.

    """
    parser = argparse.ArgumentParser(
        description="Train the KantoDex Classifier model with enhanced TensorBoard support.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/config/config.yaml",
        help="Path to the YAML configuration file. (default: src/config/config.yaml)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Flag to resume training from the latest checkpoint.",
    )
    parser.add_argument(
        "--visualize_model",
        action="store_true",
        default=False,
        help="Flag to visualize the model architecture.",
    )
    # TensorBoard CLI arguments
    parser.add_argument(
        "--enable_tensorboard",
        action="store_true",
        default=True,
        help="Enable TensorBoard logging.",
    )
    parser.add_argument(
        "--tensorboard_log_dir",
        type=str,
        default="runs/",
        help="Directory to save TensorBoard logs. Defaults to 'runs/' with timestamp.",
    )
    parser.add_argument(
        "--tensorboard_comment",
        type=str,
        default="",
        help="Comment to append to TensorBoard log directory name.",
    )
    parser.add_argument(
        "--tensorboard_purge_step",
        type=int,
        default=None,
        help="Step from which to purge events in TensorBoard.",
    )
    parser.add_argument(
        "--tensorboard_max_queue",
        type=int,
        default=10,
        help="Maximum queue size for pending TensorBoard events. (default: 10)",
    )
    parser.add_argument(
        "--tensorboard_flush_secs",
        type=int,
        default=120,
        help="How often (in seconds) to flush pending TensorBoard events to disk. (default: 120)",
    )
    parser.add_argument(
        "--tensorboard_filename_suffix",
        type=str,
        default="",
        help="Suffix for TensorBoard event filenames.",
    )

    return parser.parse_args()


def setup_logging(log_dir: str = "logs") -> None:
    """
    Set up logging to file and console.

    Args:
        log_dir (str, optional): Directory to save log files. (default: "logs")

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
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict[str, Any]: Dictionary containing configuration parameters.

    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    logging.info(f"Configuration loaded from {config_path}")
    return config


def initialize_model(config: dict[str, Any], num_classes: int, device: torch.device) -> nn.Module:
    """
    Initialize the KantoDexClassifier model based on the configuration.

    Args:
        config (dict[str, Any]): Configuration parameters.
        num_classes (int): Number of output classes.
        device (torch.device): Device to load the model on (CPU or CUDA).

    Returns:
        nn.Module: Initialized KantoDexClassifier model.

    """
    model = KantoDexClassifier(
        model_name=config["model"].get("name", "efficientnet_b3"),
        num_classes=num_classes,
        pretrained=config["model"].get("pretrained", True),
        drop_prob=config["model"].get("dropout", 0.1),
        custom_config=config["model"].get("custom_model_params", None),
    ).to(device)
    logging.info(f"Initialized model: {config['model'].get('name', 'efficientnet_b3')}")
    return model


def initialize_optimizer(config: dict[str, Any], model: nn.Module) -> optim.Optimizer:
    """
    Initialize the optimizer based on the configuration.

    Args:
        config (dict[str, Any]): Configuration parameters.
        model (nn.Module): The model to optimize.

    Returns:
        optim.Optimizer: Initialized optimizer.

    Raises:
        ValueError: If an unknown optimizer name is provided in the configuration.

    """
    optimizer_name = config["training"].get("optimizer", "adamw").lower()
    optimizer_params = config["training"]["optimizer_params"]
    if optimizer_name == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=optimizer_params.get("learning_rate", 1e-4),
            weight_decay=optimizer_params.get("weight_decay", 0.01),
            eps=optimizer_params.get("eps", 1e-8),
        )
        logging.info("Initialized AdamW optimizer.")
    elif optimizer_name == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=optimizer_params.get("learning_rate", 1e-4),
            weight_decay=optimizer_params.get("weight_decay", 0.01),
            eps=optimizer_params.get("eps", 1e-8),
        )
        logging.info("Initialized Adam optimizer.")
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=optimizer_params.get("learning_rate", 0.01),
            momentum=optimizer_params.get("momentum", 0.9),
            weight_decay=optimizer_params.get("weight_decay", 0.01),
        )
        logging.info("Initialized SGD optimizer.")
    elif optimizer_name == "rmsprop":
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=optimizer_params.get("learning_rate", 0.01),
            alpha=optimizer_params.get("alpha", 0.99),
            eps=optimizer_params.get("eps", 1e-8),
            weight_decay=optimizer_params.get("weight_decay", 0.01),
            momentum=optimizer_params.get("momentum", 0.9),
        )
        logging.info("Initialized RMSprop optimizer.")
    else:
        msg = f"Unknown optimizer: {optimizer_name}"
        logging.error(msg)
        raise ValueError(msg)
    return optimizer


def initialize_scheduler(
    config: dict[str, Any],
    optimizer: optim.Optimizer,
) -> optim.lr_scheduler._LRScheduler | None:
    """
    Initialize the learning rate scheduler based on the configuration.

    Args:
        config (dict[str, Any]): Configuration parameters.
        optimizer (optim.Optimizer): The optimizer for which to schedule the learning rate.

    Returns:
        Optional[optim.lr_scheduler._LRScheduler]: Initialized scheduler or `None` if no scheduler
        is specified.

    """
    scheduler = None
    scheduler_name = config["training"].get("scheduler", None)
    scheduler_params = config["training"].get("scheduler_params", {})

    if scheduler_name == "step_lr":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_params.get("step_size", 30),
            gamma=scheduler_params.get("gamma", 0.1),
        )
        logging.info("Initialized StepLR scheduler.")
    elif scheduler_name == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_params.get("T_max", 10),
            eta_min=scheduler_params.get("eta_min", 1e-6),
        )
        logging.info("Initialized CosineAnnealingLR scheduler.")
    elif scheduler_name == "cosine_annealing_warm_restarts":
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=scheduler_params.get("T_0", 10),
            T_mult=scheduler_params.get("T_mult", 2),
            eta_min=scheduler_params.get("eta_min", 1e-6),
        )
        logging.info("Initialized CosineAnnealingWarmRestarts scheduler.")
    elif scheduler_name == "reduce_lr_on_plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_params.get("mode", "min"),
            factor=scheduler_params.get("factor", 0.1),
            patience=scheduler_params.get("patience", 10),
            threshold=scheduler_params.get("threshold", 1e-4),
            threshold_mode=scheduler_params.get("threshold_mode", "rel"),
            cooldown=scheduler_params.get("cooldown", 0),
            min_lr=scheduler_params.get("min_lr", 0),
            eps=scheduler_params.get("eps", 1e-8),
            verbose=scheduler_params.get("verbose", False),
        )
        logging.info("Initialized ReduceLROnPlateau scheduler.")
    elif scheduler_name is not None:
        logging.warning(f"Unknown scheduler: {scheduler_name}. No scheduler will be used.")
    else:
        logging.info("No scheduler specified. Continuing without a scheduler.")

    if scheduler:
        logging.info(f"Scheduler '{scheduler_name}' has been initialized.")
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
        config (dict[str, Any]): Configuration parameters.
        model (nn.Module): The model to resume training.
        optimizer (optim.Optimizer): The optimizer.
        scheduler (Optional[optim.lr_scheduler._LRScheduler]): The learning rate scheduler.
        scaler (GradScaler): The gradient scaler for mixed precision.
        device (torch.device): Device to load the checkpoint on.

    Returns:
        Tuple[int, float]: A tuple containing the starting epoch and the best accuracy achieved
        so far.

    Raises:
        FileNotFoundError: If no checkpoint is found and resume is attempted.

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
            f"Resumed training from epoch {start_epoch} with best accuracy {best_accuracy:.2f}%.",
        )
        return start_epoch, best_accuracy
    logging.warning("No checkpoint found to resume training.")
    return 0, 0.0


def train_epoch(  # noqa: PLR0913
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    metrics_calculator: MetricsCalculator,
    tensorboard_logger: TensorBoardLogger | None = None,
    use_cutmix: bool = False,
    use_mixup: bool = False,
    alpha: float = 1.0,
    epoch: int = 0,
) -> tuple[float, float]:
    """
    Train the model for one epoch with optional CutMix and MixUp augmentations.

    Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): DataLoader for training data.
        criterion (nn.Module): Loss function.
        optimizer (optim.Optimizer): Optimizer.
        device (torch.device): Device to train on (CPU or CUDA).
        scaler (GradScaler): Gradient scaler for mixed precision training.
        metrics_calculator (MetricsCalculator): Instance to calculate training metrics.
        tensorboard_logger (Optional[TensorBoardLogger], optional): Logger for TensorBoard.
            Default is `None`.
        use_cutmix (bool, optional): Flag to enable CutMix augmentation. Default is `False`.
        use_mixup (bool, optional): Flag to enable MixUp augmentation. Default is `False`.
        alpha (float, optional): Alpha parameter for CutMix and MixUp. Default is `1.0`.
        epoch (int, optional): Current epoch number for logging purposes. Default is `0`.

    Returns:
        Tuple[float, float]: A tuple containing the average loss and average accuracy for the epoch.

    """
    model.train()
    running_loss = 0.0

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training", unit="batch")):
        images, labels = batch

        if use_cutmix or use_mixup:
            images, labels = collate_fn(images, labels, use_cutmix, use_mixup, alpha)

        images = images.to(device)
        labels = labels.to(device).long()

        optimizer.zero_grad()

        with autocast(device.type, enabled=device.type == "cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        metrics_calculator.update(outputs, labels)

        # Log batch-wise loss every 10 batches
        if tensorboard_logger and (batch_idx % 10 == 0):
            global_step = epoch * len(dataloader) + batch_idx
            tensorboard_logger.add_scalar(
                "Batch Loss",
                loss.item(),
                global_step=global_step,
            )

    avg_loss = running_loss / len(dataloader)
    metrics = metrics_calculator.compute()

    if tensorboard_logger:
        tensorboard_logger.add_scalar("Epoch Training Loss", avg_loss, epoch)
        tensorboard_logger.add_scalar("Epoch Training Accuracy", metrics["accuracy"], epoch)

    return avg_loss, metrics["accuracy"]


def validate(  # noqa: PLR0913
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    metrics_calculator: MetricsCalculator,
    tensorboard_logger: TensorBoardLogger | None = None,
    epoch: int = 0,
) -> float:
    """
    Validate the model on the validation dataset.

    Args:
        model (nn.Module): The model to validate.
        dataloader (DataLoader): DataLoader for validation data.
        device (torch.device): Device to perform validation on (CPU or CUDA).
        metrics_calculator (MetricsCalculator): Instance to calculate validation metrics.
        tensorboard_logger (Optional[TensorBoardLogger], optional): Logger for TensorBoard.
            Default is `None`.
        epoch (int, optional): Current epoch number for logging purposes. Default is `0`.

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

    if tensorboard_logger:
        tensorboard_logger.add_scalar("Epoch Validation Accuracy", metrics["accuracy"], epoch)

    return metrics["accuracy"]


def main() -> None:  # noqa: PLR0915, C901, PLR0912
    print("Training the model...")
    args = parse_args()
    setup_logging()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load data to determine number of classes before initializing the model
    train_loader_dummy, val_loader_dummy, label_to_idx_dummy = load_data(
        processed_path=config["data"]["processed_path"],
        test_size=config["data"]["test_size"],
        batch_size=1,
        img_size=tuple(config["data"]["img_size"]),
        num_workers=config["data"]["num_workers"],
        use_cutmix=False,
        use_mixup=False,
        alpha=1.0,
    )
    num_classes = len(label_to_idx_dummy)
    logging.info(f"Number of classes: {num_classes}")

    # Initialize model
    model = initialize_model(config, num_classes, device)

    # Initialize TensorBoard Logger
    tensorboard_logger = None
    if args.enable_tensorboard:
        tensorboard_logger = TensorBoardLogger(
            log_dir=args.tensorboard_log_dir,
            comment=args.tensorboard_comment,
            purge_step=args.tensorboard_purge_step,
            max_queue=args.tensorboard_max_queue,
            flush_secs=args.tensorboard_flush_secs,
            filename_suffix=args.tensorboard_filename_suffix,
            enabled=True,
        )
        # Add model graph with the actual model and a sample input
        sample_images, _ = next(iter(train_loader_dummy))
        sample_images = sample_images.to(device)
        tensorboard_logger.add_graph(model, sample_images)
        logging.info("Model graph added to TensorBoard.")

    # Load actual training and validation data
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

    # Initialize model, optimizer, scheduler
    model = initialize_model(config, num_classes, device)
    optimizer = initialize_optimizer(config, model)
    scheduler = initialize_scheduler(config, optimizer)

    # Update TensorBoard graph with the actual model if logger is enabled
    if tensorboard_logger and tensorboard_logger.enabled:
        tensorboard_logger.add_graph(model, sample_images)
        logging.info("Model graph added to TensorBoard.")

    # Loss function with Label Smoothing
    label_smoothing = config["training"].get("label_smoothing", 0.1)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    metrics_calculator = MetricsCalculator(num_classes=num_classes)

    # Mixed Precision Scaler
    scaler = GradScaler(enabled=device.type == "cuda")

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
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            metrics_calculator=metrics_calculator,
            tensorboard_logger=tensorboard_logger,
            use_cutmix=config["training"].get("use_cutmix", False),
            use_mixup=config["training"].get("use_mixup", False),
            alpha=config["training"].get("alpha", 1.0),
            epoch=epoch + 1,
        )
        logging.info(
            f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%",
        )

        # Calculate additional metrics
        precision = metrics_calculator.precision.compute().item()
        recall = metrics_calculator.recall.compute().item()
        f1_score = metrics_calculator.f1.compute().item()
        logging.info(
            f"Precision: {precision:.2f}%, Recall: {recall:.2f}%, F1 Score: {f1_score:.2f}%",
        )

        # Log additional metrics to TensorBoard
        if tensorboard_logger:
            tensorboard_logger.add_scalar("Precision", precision, epoch + 1)
            tensorboard_logger.add_scalar("Recall", recall, epoch + 1)
            tensorboard_logger.add_scalar("F1 Score", f1_score, epoch + 1)

        # Scheduler step
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(train_loss)
            else:
                scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            logging.info(f"Learning Rate: {current_lr}")
            if tensorboard_logger:
                tensorboard_logger.add_scalar("Learning Rate", current_lr, epoch + 1)

        # Validate
        val_accuracy = validate(
            model=model,
            dataloader=val_loader,
            device=device,
            metrics_calculator=metrics_calculator,
            tensorboard_logger=tensorboard_logger,
            epoch=epoch + 1,
        )
        logging.info(f"Validation Accuracy: {val_accuracy:.2f}%")

        # Check for best model
        is_best = val_accuracy > best_accuracy
        if is_best:
            best_accuracy = val_accuracy
            epochs_no_improve = 0
            logging.info(f"New best accuracy: {best_accuracy:.2f}%")
            if tensorboard_logger:
                tensorboard_logger.add_scalar("Best Accuracy", best_accuracy, epoch + 1)
        else:
            epochs_no_improve += 1
            logging.info(f"No improvement in validation accuracy for {epochs_no_improve} epochs.")

        # Save checkpoint
        checkpoint_dir = Path(config["training"]["checkpoint_path"])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            checkpoint_dir=str(checkpoint_dir),
            scheduler=scheduler,
            scaler=scaler,
            is_best=is_best,
            best_accuracy=best_accuracy,
        )

        # Early Stopping
        if epochs_no_improve >= early_stopping_patience:
            logging.info("Early stopping triggered.")
            break

    logging.info("Training complete.")
    if tensorboard_logger:
        tensorboard_logger.close()


if __name__ == "__main__":
    main()
