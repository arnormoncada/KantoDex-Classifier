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
        augmentations: dict[str, bool] | None = None,
        additional_augmentations: bool = True,
        style_specific: bool = False,
        num_styles: int = 1,
    ) -> None:
        """
        Initialize the DataAugmentor with specified image size and augmentation configurations.

        Args:
            img_size (tuple): Desired image size as (height, width).
            augmentations (dict, optional): Dictionary to enable/disable specific augmentations.
            additional_augmentations (bool, optional): Flag to include advanced augmentations like CutMix and MixUp.
            style_specific (bool, optional): Flag to apply style-specific augmentations.
            num_styles (int, optional): Number of distinct art styles in the dataset.

        """
        self.img_size = img_size
        self.augmentations = augmentations or {}
        self.additional_augmentations = additional_augmentations
        self.style_specific = style_specific
        self.num_styles = num_styles  # Number of distinct art styles

        self.transform = self.build_transform()

    def build_transform(self) -> A.Compose:
        """
        Build the augmentation pipeline based on the configurations.

        Returns:
            A.Compose: Composed augmentation pipeline.

        """
        transforms_list = []

        # Resize first to standardize image size
        transforms_list.append(A.Resize(height=self.img_size[0], width=self.img_size[1]))

        if self.augmentations.get("horizontal_flip", True):
            transforms_list.append(A.HorizontalFlip(p=0.5))

        if self.augmentations.get("vertical_flip", False):
            transforms_list.append(A.VerticalFlip(p=0.1))

        if self.augmentations.get("brightness_contrast", True):
            transforms_list.append(A.RandomBrightnessContrast(p=0.3))

        if self.augmentations.get("rotation", True):
            transforms_list.append(A.Rotate(limit=30, p=0.7, border_mode=cv2.BORDER_REFLECT_101))

        if self.augmentations.get("shift_scale_rotate", True):
            transforms_list.append(
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=0.5,
                    border_mode=cv2.BORDER_REFLECT_101,
                ),
            )

        if self.augmentations.get("gauss_noise", True):
            transforms_list.append(A.GaussNoise(var_limit=(10.0, 50.0), p=0.3))

        if self.augmentations.get("blur", False):
            transforms_list.append(A.Blur(blur_limit=5, p=0.2))

        if self.augmentations.get("elastic_transform", False):
            transforms_list.append(A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3))

        if self.augmentations.get("grid_distortion", False):
            transforms_list.append(A.GridDistortion(p=0.2))

        if self.augmentations.get("random_shadows", False):
            transforms_list.append(A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), p=0.3))

        if self.augmentations.get("channel_shuffle", False):
            transforms_list.append(A.ChannelShuffle(p=0.2))

        # Advanced Augmentations
        if self.additional_augmentations and self.augmentations.get("cutout", False):
            transforms_list.append(
                A.CoarseDropout(
                    max_holes=8,
                    max_height=self.img_size[0] // 10,
                    max_width=self.img_size[1] // 10,
                    fill_value=0,
                    p=0.5,
                ),
            )

        # Note: Removed A.Normalize to prevent double normalization
        return A.Compose(transforms_list)

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
            alpha (float): Parameter for the beta distribution.

        Returns:
            Tuple[Image.Image, float]: (Combined image, lambda value)

        """
        rng = np.random.default_rng()
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
            lam (float, optional): Lambda value for blending. If None, sampled from Beta distribution.

        Returns:
            PIL.Image.Image: Blended image.

        """
        if lam is None:
            rng = np.random.default_rng()
            lam = rng.beta(1.0, 1.0)
        image1_np = np.array(image1).astype(np.float32)
        image2_np = np.array(image2).astype(np.float32)
        mixed = lam * image1_np + (1 - lam) * image2_np
        mixed = mixed.astype(np.uint8)
        return Image.fromarray(mixed, "RGB")

    def visualize_augmentations(self, image_path: str, num_samples: int = 5) -> None:
        """
        Visualize augmented images to ensure augmentations are applied correctly.

        Args:
            image_path (str): Path to the input image.
            num_samples (int, optional): Number of augmented samples to visualize.

        """
        image = Image.open(image_path).convert("RGB")
        fig, axes = plt.subplots(1, num_samples, figsize=(15, 15))
        for i in range(num_samples):
            augmented_image = self.augment(image)
            axes[i].imshow(augmented_image)
            axes[i].axis("off")
        plt.show()
