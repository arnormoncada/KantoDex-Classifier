import logging
from pathlib import Path
from typing import Any

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.augmentation.augmentor import DataAugmentor


class PokemonDataset(Dataset):
    """PokemonDataset is a custom Dataset for loading Pokémon images and labels."""

    def __init__(
        self,
        image_paths: list,
        labels: list,
        augment: bool = False,
        transform: Any | None = None,
    ) -> None:
        """
        Initialize the dataset with image paths and labels.

        Args:
            image_paths (list): List of image file paths.
            labels (list): Corresponding list of labels.
            augment (bool, optional): Whether to apply data augmentation.
            transform (Any, optional): Transformations to apply to the images.

        """
        self.image_paths = image_paths
        self.labels = labels
        self.augment = augment
        self.transform = transform
        self.augmentor = DataAugmentor() if augment else None

    def __len__(self) -> int:
        """
        Return the total number of samples.

        Returns:
            int: Number of samples.

        """
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[Any, int]:
        """
        Retrieve the image and label at the specified index.

        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple[Any, int]: (image, label)

        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            logging.exception(f"Error loading image {img_path}")
            # Return a black image in case of error
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.augmentor:
            image = self.augmentor.augment(image)
        if self.transform:
            image = self.transform(image)
        return image, label


def load_data(
    processed_path: str,
    test_size: float = 0.2,
    batch_size: int = 32,
    img_size: tuple = (224, 224),
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, dict]:
    """
    Load and prepare the training and validation data loaders.

    Args:
        processed_path (str): Path to the processed data directory.
        test_size (float, optional): Proportion of data to use for validation.
        batch_size (int, optional): Batch size for data loaders.
        img_size (tuple, optional): Desired image size as (height, width).
        num_workers (int, optional): Number of subprocesses for data loading.

    Returns:
        Tuple[DataLoader, DataLoader, dict]: (train_loader, val_loader, label_to_idx)

    """
    processed_path = Path(processed_path)
    if not processed_path.exists():
        msg = f"Processed data path {processed_path} does not exist."
        raise FileNotFoundError(msg)

    image_paths = []
    labels = []
    for label_dir in processed_path.iterdir():
        if label_dir.is_dir():
            label = label_dir.name
            for img_file in label_dir.glob("*.*"):
                if img_file.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".gif"]:
                    image_paths.append(str(img_file))
                    labels.append(label)

    if not image_paths:
        msg = f"No images found in {processed_path}"
        raise ValueError(msg)

    # Encode labels
    label_to_idx = {label: idx for idx, label in enumerate(sorted(set(labels)))}
    labels = [label_to_idx[label] for label in labels]

    # Split data
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths,
        labels,
        test_size=test_size,
        stratify=labels,
        random_state=42,
    )

    # Define transforms
    transform = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ],
    )

    # Create datasets
    train_dataset = PokemonDataset(train_paths, train_labels, augment=True, transform=transform)
    val_dataset = PokemonDataset(val_paths, val_labels, augment=False, transform=transform)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, label_to_idx
