import numpy as np
import cv2
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

BASE_FEATURE_NAMES = [
    "mean_r",
    "std_r",
    "mean_g",
    "std_g",
    "mean_b",
    "std_b",
    "lbp_0",
    "lbp_1",
    "lbp_2",
    "lbp_3",
    "lbp_4",
    "lbp_5",
    "lbp_6",
    "lbp_7",
    "lbp_8",
    "lbp_9",
]

HARALICK_FEATURE_NAMES = [
    "haralick_contrast",
    "haralick_correlation",
    "haralick_energy",
    "haralick_homogeneity",
]

def get_feature_names(use_haralick=False):
    if use_haralick:
        return BASE_FEATURE_NAMES + HARALICK_FEATURE_NAMES
    return BASE_FEATURE_NAMES

def extract_color_features(img, mask):
    features = []
    tissue_pixels = mask > 0

    for channel in range(3):
        values = img[:, :, channel][tissue_pixels]

        if values.size == 0:
            features.extend([0.0, 0.0])
        else:
            features.append(float(np.mean(values)))
            features.append(float(np.std(values)))

    return features

def extract_lbp_features(img, mask):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")

    values = lbp[mask > 0]

    if values.size == 0:
        return [0.0] * 10

    hist, _ = np.histogram(values, bins=10, range=(0, 10))
    hist = hist.astype(float)
    hist /= (hist.sum() + 1e-6)

    return hist.tolist()

def extract_haralick_features(img, mask):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    masked_gray = gray.copy()
    masked_gray[mask == 0] = 0

    glcm = graycomatrix(
        masked_gray,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True,
    )

    features = []
    for prop in ["contrast", "correlation", "energy", "homogeneity"]:
        features.append(float(graycoprops(glcm, prop)[0, 0]))

    return features

def extract_features(img, mask, use_haralick=False):
    color_features = extract_color_features(img, mask)
    lbp_features = extract_lbp_features(img, mask)

    features = color_features + lbp_features

    if use_haralick:
        haralick_features = extract_haralick_features(img, mask)
        features += haralick_features

    return np.array(features, dtype=float)