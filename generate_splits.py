import os
import numpy as np
from dataset import build_dataset
from experiments import EXPERIMENTS
from sklearn.model_selection import RepeatedStratifiedKFold

GENERATED = set()

for config in EXPERIMENTS:

    magnification = config["magnification"]
    num_normal = config["num_normal"]
    num_tumour = config["num_tumour"]

    split_key = (
        magnification,
        num_normal,
        num_tumour,
    )

    # avoid duplicate split generation
    if split_key in GENERATED:
        continue

    GENERATED.add(split_key)

    print("\nGenerating splits for:")
    print(split_key)

    X, y, used_files, image_paths = build_dataset(
        magnification=magnification,
        num_normal=num_normal,
        num_tumour=num_tumour,
        use_haralick=False,
    )

    split_dir = (
        f"cv_splits/"
        f"{magnification}_"
        f"{num_normal}_"
        f"{num_tumour}"
    )

    os.makedirs(split_dir, exist_ok=True)

    rskf = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=5,
        random_state=42,
    )

    for fold, (train_idx, test_idx) in enumerate(
        rskf.split(X, y),
        start=1,
    ):

        np.save(
            os.path.join(
                split_dir,
                f"train_idx_fold_{fold}.npy",
            ),
            train_idx,
        )

        np.save(
            os.path.join(
                split_dir,
                f"test_idx_fold_{fold}.npy",
            ),
            test_idx,
        )

print("\nAll splits generated.")
