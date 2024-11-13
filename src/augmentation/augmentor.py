import logging
import random
from typing import Any

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class DataAugmentor:
    """Apply data augmentation transformations to images with enhanced flexibility and performance."""

    def __init__(
        self,
        img_size: tuple[int, int] = (224, 224),
        augmentations: dict[str, Any] | None = None,
        additional_augmentations: bool = True,
        style_specific: bool = False,
        num_styles: int = 1,
        seed: int | None = None,
    ) -> None:
        """
        Initialize the DataAugmentor with specified image size and augmentation configurations.

        Args:
            img_size (Tuple[int, int], optional): Desired image size as (height, width). Defaults to (224, 224).
            augmentations (Dict[str, Any], optional): Dictionary to enable/disable specific augmentations and their parameters.

        Example:
                    {
                        "horizontal_flip": {"enable": True, "p": 0.5},
                        "vertical_flip": {"enable": False, "p": 0.1},
                        ...
                    }
            additional_augmentations (bool, optional): Flag to include advanced augmentations like CutOut. Defaults to True.
            style_specific (bool, optional): Flag to apply style-specific augmentations. Defaults to False.
            num_styles (int, optional): Number of distinct art styles in the dataset. Used if style_specific is True. Defaults to 1.
            seed (int, optional): Random seed for reproducibility. Defaults to None.

        """
        self.img_size = img_size
        self.augmentations = augmentations or {}
        self.additional_augmentations = additional_augmentations
        self.style_specific = style_specific
        self.num_styles = num_styles  # Number of distinct art styles
        self.seed = seed

        # Set random seed for reproducibility if provided
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        self.transform = self.build_transform()

    def build_transform(self) -> A.Compose:
        """
        Build the augmentation pipeline based on the configurations.

        Returns:
            A.Compose: Composed augmentation pipeline.

        """
        transforms_list = []

        # Random Order to shuffle the augmentation operations
        transforms_list.append(
            A.OneOf(
                [
                    A.HorizontalFlip(p=self.augmentations.get("horizontal_flip", {}).get("p", 0.5)),
                    A.VerticalFlip(p=self.augmentations.get("vertical_flip", {}).get("p", 0.1)),
                ],
                p=0.7,
            ),
        )

        transforms_list.append(
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        p=self.augmentations.get("brightness_contrast", {}).get("p", 0.3),
                    ),
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
                ],
                p=0.5,
            ),
        )

        transforms_list.append(
            A.Rotate(
                limit=self.augmentations.get("rotation", {}).get("limit", 30),
                p=self.augmentations.get("rotation", {}).get("p", 0.7),
                border_mode=cv2.BORDER_REFLECT_101,
            ),
        )

        transforms_list.append(
            A.ShiftScaleRotate(
                shift_limit=self.augmentations.get("shift_scale_rotate", {}).get(
                    "shift_limit",
                    0.1,
                ),
                scale_limit=self.augmentations.get("shift_scale_rotate", {}).get(
                    "scale_limit",
                    0.1,
                ),
                rotate_limit=self.augmentations.get("shift_scale_rotate", {}).get(
                    "rotate_limit",
                    15,
                ),
                p=self.augmentations.get("shift_scale_rotate", {}).get("p", 0.5),
                border_mode=cv2.BORDER_REFLECT_101,
            ),
        )

        transforms_list.append(
            A.OneOf(
                [
                    A.GaussNoise(
                        var_limit=self.augmentations.get("gauss_noise", {}).get(
                            "var_limit",
                            (10.0, 50.0),
                        ),
                        p=self.augmentations.get("gauss_noise", {}).get("p", 0.3),
                    ),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.15), p=0.3),
                ],
                p=0.4,
            ),
        )

        if self.augmentations.get("blur", {}).get("enable", False):
            transforms_list.append(
                A.OneOf(
                    [
                        A.Blur(
                            blur_limit=self.augmentations.get("blur", {}).get("blur_limit", 5),
                            p=0.5,
                        ),
                        A.MotionBlur(
                            blur_limit=self.augmentations.get("blur", {}).get("blur_limit", 5),
                            p=0.5,
                        ),
                    ],
                    p=self.augmentations.get("blur", {}).get("p", 0.2),
                ),
            )

        if self.augmentations.get("elastic_transform", {}).get("enable", False):
            transforms_list.append(
                A.ElasticTransform(
                    alpha=self.augmentations.get("elastic_transform", {}).get("alpha", 1),
                    sigma=self.augmentations.get("elastic_transform", {}).get("sigma", 50),
                    alpha_affine=self.augmentations.get("elastic_transform", {}).get(
                        "alpha_affine",
                        50,
                    ),
                    p=self.augmentations.get("elastic_transform", {}).get("p", 0.3),
                ),
            )

        if self.augmentations.get("grid_distortion", {}).get("enable", False):
            transforms_list.append(
                A.GridDistortion(p=self.augmentations.get("grid_distortion", {}).get("p", 0.2)),
            )

        if self.augmentations.get("random_shadows", {}).get("enable", False):
            transforms_list.append(
                A.RandomShadow(
                    shadow_roi=self.augmentations.get("random_shadows", {}).get(
                        "shadow_roi",
                        (0, 0.5, 1, 1),
                    ),
                    p=self.augmentations.get("random_shadows", {}).get("p", 0.3),
                ),
            )

        if self.augmentations.get("channel_shuffle", {}).get("enable", False):
            transforms_list.append(
                A.ChannelShuffle(p=self.augmentations.get("channel_shuffle", {}).get("p", 0.2)),
            )

        # Advanced Augmentations
        if self.additional_augmentations and self.augmentations.get("cutout", {}).get(
            "enable",
            False,
        ):
            transforms_list.append(
                A.CoarseDropout(
                    max_holes=self.augmentations.get("cutout", {}).get("max_holes", 8),
                    max_height=self.augmentations.get("cutout", {}).get(
                        "max_height",
                        self.img_size[0] // 10,
                    ),
                    max_width=self.augmentations.get("cutout", {}).get(
                        "max_width",
                        self.img_size[1] // 10,
                    ),
                    fill_value=self.augmentations.get("cutout", {}).get("fill_value", 0),
                    p=self.augmentations.get("cutout", {}).get("p", 0.5),
                ),
            )

        # Style-Specific Augmentations
        if self.style_specific and self.num_styles > 1:
            # Define different augmentation pipelines for different styles
            style_transforms = []
            for style in range(self.num_styles):
                style_aug = [
                    A.RandomResizedCrop(
                        height=self.img_size[0],
                        width=self.img_size[1],
                        scale=(0.8, 1.0),
                        p=0.5,
                    ),
                    A.OneOf(
                        [
                            A.HorizontalFlip(p=0.5),
                            A.VerticalFlip(p=0.5),
                        ],
                        p=0.5,
                    ),
                    A.OneOf(
                        [
                            A.RandomBrightnessContrast(p=0.5),
                            A.ColorJitter(
                                brightness=0.3,
                                contrast=0.3,
                                saturation=0.3,
                                hue=0.2,
                                p=0.5,
                            ),
                        ],
                        p=0.5,
                    ),
                ]
                style_transforms.extend(style_aug)
            transforms_list.extend(style_transforms)

        # Random Order for additional randomness
        transforms_list = [A.OneOf(transforms_list, p=1.0)]

        return A.Compose(transforms_list, additional_targets={"image2": "image"})

    def augment(self, image: Image.Image) -> Image.Image:
        """
        Apply augmentation to the given image.

        Args:
            image (PIL.Image.Image): Input image.

        Returns:
            PIL.Image.Image: Augmented image.

        """
        image_np = np.array(image)

        # Define constant for RGB channels
        RGB_CHANNELS = 3

        # Convert RGB to BGR for OpenCV compatibility if needed
        if image_np.shape[2] == RGB_CHANNELS:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        augmented = self.transform(image=image_np)
        augmented_image = augmented["image"]

        # Convert BGR back to RGB
        if augmented_image.shape[2] == RGB_CHANNELS:
            augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB)

        return Image.fromarray(augmented_image.astype("uint8"), "RGB")

    def apply_cutmix(
        self,
        image1: Image.Image,
        image2: Image.Image,
        alpha: float = 1.0,
    ) -> tuple[Image.Image, float]:
        """
        Apply CutMix augmentation by combining two images.

        Args:
            image1 (PIL.Image.Image): First image.
            image2 (PIL.Image.Image): Second image.
            alpha (float, optional): Parameter for the beta distribution. Defaults to 1.0.

        Returns:
            Tuple[Image.Image, float]: (Combined image, lambda value)

        """
        rng = np.random.default_rng(self.seed)
        lam = rng.beta(alpha, alpha)
        w, h = self.img_size
        cut_w = int(w * np.sqrt(1 - lam))
        cut_h = int(h * np.sqrt(1 - lam))

        # Uniformly sample the center of the patch
        cx = rng.integers(w)
        cy = rng.integers(h)

        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)

        image1_np = np.array(image1).astype(np.uint8)
        image2_np = np.array(image2).astype(np.uint8)

        # Apply CutMix
        image1_np[bby1:bby2, bbx1:bbx2, :] = image2_np[bby1:bby2, bbx1:bbx2, :]

        combined_image = Image.fromarray(image1_np, "RGB")
        return combined_image, lam

    def apply_mixup(
        self,
        image1: Image.Image,
        image2: Image.Image,
        lam: float | None = None,
    ) -> Image.Image:
        """
        Apply MixUp augmentation by blending two images.

        Args:
            image1 (PIL.Image.Image): First image.
            image2 (PIL.Image.Image): Second image.
            lam (float, optional): Lambda value for blending. If None, sampled from Beta distribution. Defaults to None.

        Returns:
            PIL.Image.Image: Blended image.

        """
        if lam is None:
            rng = np.random.default_rng(self.seed)
            lam = rng.beta(1.0, 1.0)
        image1_np = np.array(image1).astype(np.float32)
        image2_np = np.array(image2).astype(np.float32)
        mixed = lam * image1_np + (1 - lam) * image2_np
        mixed = np.clip(mixed, 0, 255).astype(np.uint8)
        return Image.fromarray(mixed, "RGB")

    def visualize_augmentations(self, image_path: str, num_samples: int = 5) -> None:
        """
        Visualize augmented images to ensure augmentations are applied correctly.

        Args:
            image_path (str): Path to the input image.
            num_samples (int, optional): Number of augmented samples to visualize. Defaults to 5.

        """
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logging.exception(f"Error loading image {image_path}: {e}")
            return

        fig, axes = plt.subplots(1, num_samples, figsize=(15, 15))
        for i in range(num_samples):
            augmented_image = self.augment(image)
            axes[i].imshow(augmented_image)
            axes[i].axis("off")
        plt.tight_layout()
        plt.show()
