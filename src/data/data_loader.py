import logging
import random
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.augmentation.augmentor import DataAugmentor


class PokemonDataset(Dataset):
    """Custom Dataset for loading Pokémon images and labels."""

    def __init__(
        self,
        image_paths: list[str],
        labels: list[int],
        augment: bool = False,
        transform: Callable | None = None,
        augmentor: DataAugmentor | None = None,
    ) -> None:
        """
        Initialize the dataset with image paths and labels.

        Args:
            image_paths (List[str]): List of image file paths.
            labels (List[int]): Corresponding list of labels.
            augment (bool, optional): Whether to apply data augmentation.
            transform (Callable, optional): Transformations to apply to the images.
            augmentor (DataAugmentor, optional): DataAugmentor instance for augmentations.

        """
        self.image_paths = image_paths
        self.labels = labels
        self.augment = augment
        self.transform = transform
        self.augmentor = augmentor if augmentor else (DataAugmentor() if augment else None)

    def __len__(self) -> int:
        """
        Return the total number of samples.

        Returns:
            int: Number of samples.

        """
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """
        Retrieve the image and label at the specified index.

        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple[torch.Tensor, int]: (image, label)

        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(img_path).convert("RGBA")
            # Convert to RGB if the image has transparency
            if image.mode == "RGBA":
                background = Image.new("RGB", image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[3])  # 3 is the alpha channel
                image = background
            else:
                image = image.convert("RGB")
        except Exception as e:
            logging.exception(f"Error loading image {img_path}: {e}")
            # Return a black image in case of error
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.augmentor and self.augment:
            image = self.augmentor.augment(image)

        if self.transform:
            image = self.transform(image)

        return image, label


def collate_fn(batch: list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor]:
    images, labels = zip(*batch, strict=False)
    images = torch.stack(images)
    labels = torch.tensor(labels, dtype=torch.long)
    return images, labels


def load_data(
    processed_path: str,
    test_size: float = 0.2,
    batch_size: int = 32,
    img_size: tuple[int, int] = (224, 224),
    num_workers: int = 4,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, dict[str, int]]:
    """
    Load and prepare the training and validation data loaders.

    Args:
        processed_path (str): Path to the processed data directory.
        test_size (float, optional): Proportion of data to use for validation.
        batch_size (int, optional): Batch size for data loaders.
        img_size (Tuple[int, int], optional): Desired image size as (height, width).
        num_workers (int, optional): Number of subprocesses for data loading.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        Tuple[DataLoader, DataLoader, Dict[str, int]]: (train_loader, val_loader, label_to_idx)

    """
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    processed_path = Path(processed_path)
    if not processed_path.exists():
        msg = f"Processed data path {processed_path} does not exist."
        logging.error(msg)
        raise FileNotFoundError(msg)

    image_paths = []
    labels = []
    for label_dir in processed_path.iterdir():
        if label_dir.is_dir():
            label = label_dir.name
            imgs = list(label_dir.glob("*.*"))
            for img_file in imgs:
                if img_file.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".gif"}:
                    image_paths.append(str(img_file))
                    labels.append(label)

    if not image_paths:
        msg = f"No images found in {processed_path}"
        logging.error(msg)
        raise ValueError(msg)

    # Encode labels using LabelEncoder for consistency
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    label_to_idx = {label: idx for idx, label in enumerate(label_encoder.classes_)}

    # Split data
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths,
        labels_encoded,
        test_size=test_size,
        stratify=labels_encoded,
        random_state=seed,
    )

    # Define transforms
    transform_train = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ],
    )

    transform_val = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ],
    )
    train_dataset = PokemonDataset(
        image_paths=train_paths,
        labels=train_labels,
        augment=True,
        transform=transform_train,
        augmentor=None,
    )
    val_dataset = PokemonDataset(
        image_paths=val_paths,
        labels=val_labels,
        augment=False,
        transform=transform_val,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        collate_fn=collate_fn,
    )

    logging.info(
        f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples.",
    )
    logging.info(f"Number of classes: {len(label_to_idx)}")

    return train_loader, val_loader, label_to_idx
