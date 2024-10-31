import albumentations as A
import numpy as np
from PIL import Image


class DataAugmentor:
    """Apply data augmentation transformations to images."""

    def __init__(self, img_size: tuple = (224, 224)) -> None:
        """
        Initialize the DataAugmentor with specified image size.

        Args:
            img_size (tuple): Desired image size as (height, width).

        """
        self.transform = A.Compose(
            [
                A.RandomResizedCrop(
                    height=img_size[0],
                    width=img_size[1],
                    scale=(0.8, 1.0),
                ),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.Rotate(limit=15, p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=0.5,
                ),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
                A.Blur(blur_limit=3, p=0.1),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ],
        )

    def augment(self, image: Image.Image) -> Image.Image:
        """
        Apply augmentation to the given image.

        Args:
            image (PIL.Image.Image): Input image.

        Returns:
            PIL.Image.Image: Augmented image.

        """
        image_np = np.array(image)
        augmented = self.transform(image=image_np)
        return Image.fromarray(augmented["image"].astype("uint8"), "RGB")
