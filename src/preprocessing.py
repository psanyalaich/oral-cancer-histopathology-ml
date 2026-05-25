import cv2
import numpy as np

from configs.preprocessing_config import (
    IMAGE_SIZE,
    MIN_TISSUE_FRACTION,
    MAX_TISSUE_FRACTION,
    GAUSSIAN_KERNEL_SIZE,
    MORPH_KERNEL_SIZE,
    MIN_COMPONENT_AREA_RATIO,
    DARK_IMAGE_THRESHOLD
)

def load_image(path, size=IMAGE_SIZE):
    img = cv2.imread(path)

    if img is None:
        raise ValueError(f"Could not load image: {path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)

    return img

def segment_tissue(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(
        gray,
        GAUSSIAN_KERNEL_SIZE,
        0
    )

    if gray.mean() < DARK_IMAGE_THRESHOLD:
        raise ValueError(
            "Image too dark for reliable segmentation."
        )

    _, mask = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        MORPH_KERNEL_SIZE
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    cleaned_mask = np.zeros_like(mask)

    min_component_area = int(
        MIN_COMPONENT_AREA_RATIO * mask.shape[0] * mask.shape[1]
    )

    for i in range(1, num_labels):

        area = stats[i, cv2.CC_STAT_AREA]

        if area >= min_component_area:
            cleaned_mask[labels == i] = 255

    mask = cleaned_mask

    tissue_fraction = np.count_nonzero(mask) / mask.size

    if tissue_fraction < MIN_TISSUE_FRACTION:
        raise ValueError(
            "Segmentation failed: insufficient tissue detected."
        )

    if tissue_fraction > MAX_TISSUE_FRACTION:
        raise ValueError(
            "Segmentation failed: excessive tissue detected."
        )

    return mask

def normalize_staining(
    img,
    method=None,
    target_image=None
):

    if method is None:
        return img

    if target_image is None:
        raise ValueError(
            "Target image required for stain normalization."
        )
    elif method == "reinhard":
        raise NotImplementedError(
            "Reinhard stain normalization not implemented yet."
        )
    else:
        raise ValueError(
            f"Unknown normalization method: {method}"
        )
