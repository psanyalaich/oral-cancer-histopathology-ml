import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from skimage.feature import graycomatrix, graycoprops

# CONFIG
IMAGE_SIZE = (512, 512)
LBP_RADIUS = 1
LBP_POINTS = 8 * LBP_RADIUS
MIN_TISSUE_FRACTION = 0.05

# IMAGE LOAD
def load_image(path):
    img = cv2.imread(path)

    if img is None:
        raise ValueError(f"Could not load image: {path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# PREPROCESSING
def preprocess_image(img):
    resized = cv2.resize(img, IMAGE_SIZE)
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)

    _, mask = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones((3, 3), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    tissue = resized.copy()
    tissue[mask == 0] = 255

    return resized, gray, mask, tissue

# EDGE MAP
def compute_edges(gray):

    edges = cv2.Canny(
        gray,
        threshold1 = 80,
        threshold2 = 160
    )

    return edges

# LBP FEATURES
def compute_lbp(gray):

    lbp = local_binary_pattern(
        gray,
        P = LBP_POINTS,
        R = LBP_RADIUS,
        method = "uniform"
    )

    return lbp

# HARALICK FEATURES
def compute_haralick(gray, mask):

    tissue_fraction = np.mean(mask > 0)

    if tissue_fraction < MIN_TISSUE_FRACTION:

        return {
            "contrast": 0.0,
            "correlation": 0.0,
            "energy": 0.0,
            "homogeneity": 0.0
        }

    masked = gray.copy()

    masked[mask == 0] = 0

    quantized = (masked // 16).astype(np.uint8)

    glcm = graycomatrix(
        quantized,
        distances = [1],
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels = 16,
        symmetric = True,
        normed = True
    )

    features = {}

    for prop in [
        "contrast",
        "correlation",
        "energy",
        "homogeneity"
    ]:

        features[prop] = graycoprops(
            glcm,
            prop
        ).mean()

    return features

# COLOUR FEATURES
def compute_color_features(img, mask):

    pixels = img[mask > 0]

    if len(pixels) == 0:
        return {}

    features = {

        "mean_r": np.mean(pixels[:, 0]),
        "mean_g": np.mean(pixels[:, 1]),
        "mean_b": np.mean(pixels[:, 2]),

        "std_r": np.std(pixels[:, 0]),
        "std_g": np.std(pixels[:, 1]),
        "std_b": np.std(pixels[:, 2])
    }

    return features

# MAIN PIPELINE
def show_pipeline(image_path, title="IMAGE"):

    print("\n" + "=" * 60)
    print(f"PROCESSING: {title}")
    print("=" * 60)

    img = load_image(image_path)
    resized, gray, mask, tissue = preprocess_image(img)
    edges = compute_edges(gray)
    haralick = compute_haralick(gray, mask)

    color_features = compute_color_features(
        resized,
        mask
    )

    print("\nColor Features:")

    for k, v in color_features.items():
        print(f"{k}: {v:.4f}")

    print("\nHaralick Features:")

    for k, v in haralick.items():
        print(f"{k}: {v:.4f}")

    feature_text = (
        f"Haralick Features\n\n"
        f"Contrast      : {haralick['contrast']:.4f}\n"
        f"Correlation   : {haralick['correlation']:.4f}\n"
        f"Energy        : {haralick['energy']:.4f}\n"
        f"Homogeneity   : {haralick['homogeneity']:.4f}\n\n"
        f"Color Features\n\n"
        f"Mean R        : {color_features['mean_r']:.2f}\n"
        f"Mean G        : {color_features['mean_g']:.2f}\n"
        f"Mean B        : {color_features['mean_b']:.2f}"
    )

    fig, axes = plt.subplots(
        2,
        3,
        figsize = (16, 10)
    )

    fig.suptitle(
        title,
        fontsize = 22,
        fontweight = 'bold'
    )

    # Original image
    axes[0, 0].imshow(resized)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    # Grayscale
    axes[0, 1].imshow(gray, cmap = "gray")
    axes[0, 1].set_title("Grayscale")
    axes[0, 1].axis("off")

    # Tissue mask
    axes[0, 2].imshow(mask, cmap = "gray")
    axes[0, 2].set_title("Tissue Mask")
    axes[0, 2].axis("off")

    # Masked tissue
    axes[1, 0].imshow(tissue)
    axes[1, 0].set_title("Masked Tissue")
    axes[1, 0].axis("off")

    # Edge map
    axes[1, 1].imshow(edges, cmap = "gray")
    axes[1, 1].set_title("Canny Edge Map")
    axes[1, 1].axis("off")

    # Feature summary
    axes[1, 2].text(
        0.02,
        0.98,
        feature_text,
        fontsize = 11,
        verticalalignment = 'top',
        family = 'monospace'
    )

    axes[1, 2].set_title("Extracted Features")
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.show()

# MAIN
if __name__ == "__main__":

    # 100x normal
    show_pipeline(
        "data/demo_oscc/Normal_100x_45.jpg",
        "100x NORMAL"
    )

    # 100 tumour
    show_pipeline(
        "data/demo_oscc/OSCC_100x_363.jpg",
        "100x TUMOUR"
    )

    # 400x normal
    show_pipeline(
        "data/demo_oscc/Normal_400x_153.jpg",
        "400x NORMAL"
    )

    # 400x tumour
    show_pipeline(
        "data/demo_oscc/OSCC_400x_15.jpg",
        "400x TUMOUR"
    )
