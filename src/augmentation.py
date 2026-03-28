"""On-the-fly image augmentation for nadir aerial imagery.

Nadir (top-down) satellite images are orientation-invariant, so we can apply
random rotation and flip to produce up to 8 distinct orientations per image.
"""

import random

from PIL import Image


def augment_image(image: Image.Image) -> Image.Image:
    """Random rotation (0/90/180/270) + flip (H/V) for nadir aerial images."""
    k = random.randint(0, 3)  # 0, 90, 180, 270 degrees
    if k:
        image = image.rotate(-90 * k, expand=True)
    if random.random() > 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() > 0.5:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
    return image
