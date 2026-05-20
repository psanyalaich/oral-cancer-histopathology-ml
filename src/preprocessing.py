import cv2
import numpy as np

MIN_TISSUE_FRACTION = 0.05
MAX_TISSUE_FRACTION = 0.95

def load_image(path, size=(512, 512)):
    img = cv2.imread(path)

    if img is None:
        raise ValueError(f"Could not load image: {path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)

    return img

def segment_tissue(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # DARK IMAGE SAFEGUARD
    if gray.mean() < 15:
        raise ValueError(
            "Image too dark for reliable segmentation."
        )

    _, mask = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7),
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

    # REMOVE SMALL NOISE
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    cleaned_mask = np.zeros_like(mask)

    min_component_area = int(
        0.001 * mask.shape[0] * mask.shape[1]
    )

    for i in range(1, num_labels):

        area = stats[i, cv2.CC_STAT_AREA]

        if area >= min_component_area:
            cleaned_mask[labels == i] = 255

    mask = cleaned_mask

    tissue_fraction = np.count_nonzero(mask) / mask.size

    # EMPTY OR FAILED MASK CHECK
    if tissue_fraction < MIN_TISSUE_FRACTION:

        raise ValueError(
            "Segmentation failed: insufficient tissue detected."
        )

    # OVERSEGMENTATION CHECK
    if tissue_fraction > MAX_TISSUE_FRACTION:

        raise ValueError(
            "Segmentation failed: excessive tissue detected."
        )

    return mask