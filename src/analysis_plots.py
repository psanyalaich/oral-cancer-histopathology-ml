import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_class_distribution(y, save_path, title = "Class Distribution"):
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok = True)

    counts = np.bincount(y.astype(int), minlength = 2)

    plt.figure(figsize = (5, 4))
    plt.bar(["Normal", "Tumour"], counts)
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi = 300, bbox_inches = "tight")
    plt.close()


def plot_feature_correlation(X, feature_names, save_path, title = "Feature Correlation Heatmap"):
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok = True)

    corr = np.corrcoef(X, rowvar = False)

    plt.figure(figsize = (14, 12))
    im = plt.imshow(corr, cmap = "coolwarm", vmin = -1, vmax = 1)
    plt.colorbar(im, fraction = 0.046, pad = 0.04)
    plt.xticks(range(len(feature_names)), feature_names, rotation = 90, fontsize = 8)
    plt.yticks(range(len(feature_names)), feature_names)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi = 300, bbox_inches = "tight")
    plt.close()
