import os
import numpy as np
from src.preprocessing import load_image, segment_tissue
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

def _limit_files(files, limit):
    if limit is None:
        return files
    return files[:limit]

def build_dataset(
    magnification="100x",
    num_normal=None,
    num_tumour=None,
    use_haralick=False,
):
    normal_dir = f"data/{magnification}/normal"
    tumour_dir = f"data/{magnification}/tumour"

    normal_files = _limit_files(_list_image_files(normal_dir), num_normal)
    tumour_files = _limit_files(_list_image_files(tumour_dir), num_tumour)

    print(f"Using {len(normal_files)} normal images")
    print(f"Using {len(tumour_files)} tumour images")

    X = []
    y = []

    for filename in normal_files:
        path = os.path.join(normal_dir, filename)
        try:
            img = load_image(path)
            mask = segment_tissue(img)
            features = extract_features(img, mask, use_haralick=use_haralick)

            X.append(features)
            y.append(0)
        except Exception as e:
            print(f"Failed NORMAL: {filename}")
            print(e)

    for filename in tumour_files:
        path = os.path.join(tumour_dir, filename)
        try:
            img = load_image(path)
            mask = segment_tissue(img)
            features = extract_features(img, mask, use_haralick=use_haralick)

            X.append(features)
            y.append(1)
        except Exception as e:
            print(f"Failed TUMOUR: {filename}")
            print(e)

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    if len(X) == 0:
        raise ValueError("No images were loaded.")

    return X, y