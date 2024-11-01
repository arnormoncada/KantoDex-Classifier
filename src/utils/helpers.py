import logging
from pathlib import Path

import torch
from torch import nn, optim
from torch.amp import GradScaler
from torch.optim.lr_scheduler import _LRScheduler


def save_checkpoint(  # noqa: PLR0913
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    checkpoint_dir: str,
    scheduler: _LRScheduler | None = None,
    scaler: GradScaler | None = None,
    is_best: bool = False,
    best_accuracy: float | None = None,
) -> None:
    """
    Save the model checkpoint.

    Args:
        model (nn.Module): The model to save.
        optimizer (optim.Optimizer): The optimizer.
        epoch (int): Current epoch number.
        checkpoint_dir (str): Directory to save checkpoints.
        scheduler (_LRScheduler, optional): Learning rate scheduler.
        scaler (GradScaler, optional): Gradient scaler for mixed precision.
        is_best (bool, optional): Flag indicating if this is the best model so far.
        best_accuracy (float, optional): Best validation accuracy.

    """
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "best_accuracy": best_accuracy,
    }
    if scheduler:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    checkpoint_file = checkpoint_path / f"checkpoint_epoch_{epoch + 1}.pth"
    torch.save(checkpoint, checkpoint_file)
    logging.info(f"Checkpoint saved at {checkpoint_file}")

    if is_best and best_accuracy is not None:
        best_path = checkpoint_path / "best_model.pth"
        torch.save(checkpoint, best_path)
        logging.info(f"Best model updated at {best_path} with accuracy: {best_accuracy:.2f}%")


def create_directory(path: str) -> None:
    """
    Create a directory if it doesn't exist.

    Args:
        path (str): Path to the directory.

    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    logging.info(f"Directory ensured at {directory}")
