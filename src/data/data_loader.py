import logging
import random
from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.augmentation.augmentor import DataAugmentor


class PokemonDataset(Dataset):
    """PokemonDataset is a custom Dataset for loading Pokémon images and labels."""

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
            with open(img_path, "rb") as f:
                image = Image.open(f).convert("RGBA")
                # Convert to RGB if the image has transparency
                if image.mode == "RGBA":
                    background = Image.new("RGB", image.size, (255, 255, 255))
                    background.paste(image, mask=image.split()[3])  # 3 is the alpha channel
                    image = background
                else:
                    image = image.convert("RGB")
        except Exception:
            logging.exception(f"Error loading image {img_path}")
            # Return a black image in case of error
            image = Image.new("RGB", (224, 224), (0, 0, 0))
        if self.augmentor:
            image = self.augmentor.augment(image)
        if self.transform:
            image = self.transform(image)
        return image, label


def collate_fn(
    batch: list[tuple[torch.Tensor, int]],
    use_cutmix: bool = False,
    use_mixup: bool = False,
    alpha: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply CutMix and MixUp augmentations at the batch level.

    Args:
        batch (List[Tuple[torch.Tensor, int]]): List of tuples containing images and labels.
        use_cutmix (bool, optional): Whether to apply CutMix augmentation.
        use_mixup (bool, optional): Whether to apply MixUp augmentation.
        alpha (float, optional): Parameter for the beta distribution used in CutMix and MixUp.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Batch of images and adjusted labels.

    """
    images, labels = zip(*batch, strict=False)
    # Resize images to the same size
    images = [transforms.functional.resize(img, (224, 224)) for img in images]
    images = torch.stack(images)
    labels = torch.tensor(labels)

    CUTMIX_PROBABILITY = 0.5
    rng = np.random.default_rng()

    if use_cutmix and random.random() < CUTMIX_PROBABILITY:
        lam = rng.beta(alpha, alpha)
        batch_size, C, H, W = images.size()
        index = torch.randperm(batch_size)
        shuffled_images = images[index]
        shuffled_labels = labels[index]

        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # Uniformly sample the center of the patch
        cx = rng.integers(W)
        cy = rng.integers(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        images[:, :, bby1:bby2, bbx1:bbx2] = shuffled_images[:, :, bby1:bby2, bbx1:bbx2]

        # Adjust lambda based on the actual area of the patch
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        labels = lam * labels.float() + (1 - lam) * shuffled_labels.float()

    elif use_mixup and random.random() < CUTMIX_PROBABILITY:
        lam = rng.beta(alpha, alpha)
        batch_size = images.size(0)
        index = torch.randperm(batch_size)
        shuffled_images = images[index]
        shuffled_labels = labels[index]

        images = lam * images + (1 - lam) * shuffled_images
        labels = lam * labels.float() + (1 - lam) * shuffled_labels.float()
    else:
        labels = labels.float()

    return images, labels


def load_data(  # noqa: PLR0913
    processed_path: str,
    test_size: float = 0.2,
    batch_size: int = 32,
    img_size: tuple = (224, 224),
    num_workers: int = 4,
    use_cutmix: bool = False,
    use_mixup: bool = False,
    alpha: float = 1.0,
) -> tuple[DataLoader, DataLoader, dict[str, int]]:
    """
    Load and prepare the training and validation data loaders.

    Args:
        processed_path (str): Path to the processed data directory.
        test_size (float, optional): Proportion of data to use for validation.
        batch_size (int, optional): Batch size for data loaders.
        img_size (tuple, optional): Desired image size as (height, width).
        num_workers (int, optional): Number of subprocesses for data loading.
        use_cutmix (bool, optional): Whether to apply CutMix augmentation.
        use_mixup (bool, optional): Whether to apply MixUp augmentation.
        alpha (float, optional): Parameter for the beta distribution used in CutMix and MixUp.

    Returns:
        Tuple[DataLoader, DataLoader, Dict[str, int]]: (train_loader, val_loader, label_to_idx)

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

    # Define transforms: Only ToTensor and Normalize
    transform_train = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ],
    )

    transform_val = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ],
    )

    augmentor = DataAugmentor(img_size=img_size) if use_cutmix or use_mixup else None

    train_dataset = PokemonDataset(
        train_paths,
        train_labels,
        augment=True,
        transform=transform_train,
        augmentor=augmentor,
    )
    val_dataset = PokemonDataset(val_paths, val_labels, augment=False, transform=transform_val)

    partial_collate_fn = partial(
        collate_fn,
        use_cutmix=use_cutmix,
        use_mixup=use_mixup,
        alpha=alpha,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        collate_fn=partial_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        collate_fn=partial_collate_fn,
    )

    return train_loader, val_loader, label_to_idx
