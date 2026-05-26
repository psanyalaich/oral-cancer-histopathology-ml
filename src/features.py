import cv2
import numpy as np

from skimage.feature import(
    local_binary_pattern, 
    graycomatrix, 
    graycoprops
)

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
    "lbp_9"
]

HARALICK_FEATURE_NAMES = [
    "haralick_contrast",
    "haralick_correlation",
    "haralick_energy",
    "haralick_homogeneity"
]

MIN_HARALICK_TISSUE_DENSITY = 0.70

FEATURE_GROUPS = {
    "color": ["color"],
    "lbp": ["lbp"],
    "haralick": ["haralick"],

    "color_lbp": [
        "color",
        "lbp"
    ],

    "color_haralick": [
        "color",
        "haralick"
    ],

    "lbp_haralick": [
        "lbp",
        "haralick"
    ],

    "all": [
        "color",
        "lbp",
        "haralick"
    ]
}

MIN_TISSUE_FRACTION = 0.05

def get_feature_names(feature_set="all"):

    if feature_set not in FEATURE_GROUPS:
        raise ValueError(
            f"Unknown feature set: {feature_set}"
        )

    feature_groups = FEATURE_GROUPS[feature_set]

    names = []

    if "color" in feature_groups:
        names.extend(BASE_FEATURE_NAMES[:6])

    if "lbp" in feature_groups:
        names.extend(BASE_FEATURE_NAMES[6:])

    if "haralick" in feature_groups:
        names.extend(HARALICK_FEATURE_NAMES)

    return names

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

    tissue_fraction = np.mean(mask > 0)

    if tissue_fraction < MIN_TISSUE_FRACTION:
        return [0.0] * 10

    lbp = local_binary_pattern(gray, P = 8, R = 1, method = "uniform")

    values = lbp[mask > 0]

    if values.size == 0:
        return [0.0] * 10

    hist, _ = np.histogram(values, bins = 10, range = (0, 10))
    hist = hist.astype(float)
    hist /= (hist.sum() + 1e-6)

    return hist.tolist()

def extract_haralick_features(img, mask):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    tissue_fraction = np.mean(mask > 0)

    if tissue_fraction < MIN_TISSUE_FRACTION:
        return [0.0, 0.0, 0.0, 0.0]

    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return [0.0, 0.0, 0.0, 0.0]

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    cropped_gray = gray[y_min:y_max + 1, x_min:x_max + 1]
    cropped_mask = mask[y_min:y_max + 1, x_min:x_max + 1]

    rows_t, cols_t = np.where(cropped_mask > 0)

    if len(rows_t) == 0:
        return [0.0, 0.0, 0.0, 0.0]

    y0, y1 = rows_t.min(), rows_t.max()
    x0, x1 = cols_t.min(), cols_t.max()

    tight_gray = cropped_gray[y0:y1 + 1, x0:x1 + 1]
    tight_mask = cropped_mask[y0:y1 + 1, x0:x1 + 1]

    tissue_density = np.mean(tight_mask > 0)

    if tissue_density < 0.70:
        return [0.0, 0.0, 0.0, 0.0]

    tissue_gray = tight_gray.copy()
    tissue_gray[tight_mask == 0] = 0

    tissue_gray = (tissue_gray // 16).astype(np.uint8)

    glcm = graycomatrix(
        tissue_gray,
        distances = [1],
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels = 16,
        symmetric = True,
        normed = True
    )

    features = []

    for prop in [
        "contrast",
        "correlation",
        "energy",
        "homogeneity"
    ]:

        value = graycoprops(glcm, prop).mean()
        features.append(float(value))

    return features

def extract_features(
    img,
    mask,
    feature_set="all"
):

    if feature_set not in FEATURE_GROUPS:
        raise ValueError(
            f"Unknown feature set: {feature_set}"
        )

    feature_groups = FEATURE_GROUPS[feature_set]

    features = []

    if "color" in feature_groups:
        features += extract_color_features(
            img,
            mask
        )

    if "lbp" in feature_groups:
        features += extract_lbp_features(
            img,
            mask
        )

    if "haralick" in feature_groups:
        features += extract_haralick_features(
            img,
            mask
        )

    return np.array(features, dtype=float)
