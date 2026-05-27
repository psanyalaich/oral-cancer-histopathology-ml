import pickle
from pathlib import Path
from src.dataset import build_dataset

CACHE_VERSION = "v3"

def get_or_cache_dataset(
    magnification,
    num_normal,
    num_tumour,
    feature_set,
    cache_dir="feature_cache",
    stain_normalization = None,
    seed = None
    ):

    key = (
        f"{CACHE_VERSION}_"
        f"{magnification}_"
        f"{num_normal}_"
        f"{num_tumour}_"
        f"feature_set_{feature_set}_"
        f"stain_{stain_normalization or 'no_norm'}_"
        f"seed_{seed}"
        )

    cache_path = Path(cache_dir) / f"{key}.pkl"

    if cache_path.exists():
        print(f"Loading cached dataset: {cache_path}")

        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"Building dataset and caching: {key}")

    X, y, used_files, image_paths = build_dataset(
        magnification = magnification,
        num_normal = num_normal,
        num_tumour = num_tumour,
        feature_set = feature_set,
        stain_normalization = stain_normalization,
        seed = seed
    )

    cache_path.parent.mkdir(exist_ok = True)

    with open(cache_path, "wb") as f:
        pickle.dump(
            (X, y, used_files, image_paths),
            f,
            protocol=pickle.HIGHEST_PROTOCOL
        )

    return X, y, used_files, image_paths
