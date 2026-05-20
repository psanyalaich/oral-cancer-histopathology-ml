import os
import numpy as np
import pandas as pd
from src.preprocessing import (
    load_image, 
    segment_tissue
)
from src.features import extract_features

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

def _list_image_files(folder):
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")

    files = []
    for name in sorted(os.listdir(folder)):
        ext = os.path.splitext(name)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            files.append(name)
    return files

def _limit_files(files, limit, seed=42):
    if limit is None or limit >= len(files):
        return files

    rng = np.random.default_rng(seed)

    selected_indices = rng.choice(
        len(files),
        size=limit,
        replace=False
        )

    selected_indices = sorted(selected_indices)

    return [files[i] for i in selected_indices]

def build_dataset(
    magnification="100x",
    num_normal=None,
    num_tumour=None,
    feature_set="all",
    ):
    
    normal_dir = f"data/{magnification}/normal"
    tumour_dir = f"data/{magnification}/tumour"

    normal_files = _limit_files(_list_image_files(normal_dir), num_normal)
    tumour_files = _limit_files(_list_image_files(tumour_dir), num_tumour)

    print(f"Using {len(normal_files)} normal images")
    print(f"Using {len(tumour_files)} tumour images")

    X = []
    y = []

    image_paths = []

    # NORMAL LOOP
    for filename in normal_files:
        path = os.path.join(normal_dir, filename)
        try:
            img = load_image(path)
            mask = segment_tissue(img)
            features = extract_features(img, mask, feature_set=feature_set)

            X.append(features)
            y.append(0)
            image_paths.append(path)
        except Exception as e:
            print(f"Failed NORMAL: {filename}")
            print(e)

    # TUMOUR LOOP
    for filename in tumour_files:
        path = os.path.join(tumour_dir, filename)
        try:
            img = load_image(path)
            mask = segment_tissue(img)
            features = extract_features(img, mask, feature_set=feature_set)

            X.append(features)
            y.append(1)
            image_paths.append(path)
        except Exception as e:
            print(f"Failed TUMOUR: {filename}")
            print(e)

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    if len(X) == 0:
        raise ValueError("No images were loaded.")
    
    # SAVE FILE MANIFEST
    used_files = {
        "magnification": magnification,
        "normal_files": normal_files,
        "tumour_files": tumour_files,
    }

    assert len(X) == len(image_paths)
    assert len(y) == len(image_paths)

    dataset_index = pd.DataFrame({
        "index": np.arange(len(image_paths)),
        "image_path": image_paths,
        "label": y,
    })

    dataset_index.to_csv(
        f"data/{magnification}/dataset_index.csv",
        index=False,
    )

    return X, y, used_files, image_paths
