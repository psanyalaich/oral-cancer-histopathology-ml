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
    seed = config["seed"]

    split_key = (
        magnification,
        num_normal,
        num_tumour,
        seed
    )

    if split_key in GENERATED:
        continue

    GENERATED.add(split_key)

    print("\nGenerating splits for:")
    print(split_key)

    X, y, used_files, image_paths = build_dataset(
        magnification=  magnification,
        num_normal = num_normal,
        num_tumour = num_tumour,
        feature_set = "all"
    )

    split_dir = (
        f"cv_splits/"
        f"{magnification}_"
        f"{num_normal}_"
        f"{num_tumour}_"
        f"seed_{seed}"
    )

    os.makedirs(split_dir, exist_ok=True)

    rskf = RepeatedStratifiedKFold(
        n_splits = 5,
        n_repeats = 5,
        random_state = seed
    )

    for fold, (train_idx, test_idx) in enumerate(
        rskf.split(X, y),
        start = 1
    ):

        np.save(
            os.path.join(
                split_dir,
                f"train_idx_fold_{fold}.npy",
            ),
            train_idx
        )

        np.save(
            os.path.join(
                split_dir,
                f"test_idx_fold_{fold}.npy",
            ),
            test_idx
        )

print("\nAll splits generated.")













# for future work when patient ID is available (mice)

'''
1. Add - StratifiedGroupKFold, - to sklearn.model_selection import 

2. Add - groups - to X, y, used_files, image_paths = build_dataset()

        groups = [
            "mouse_01",
            "mouse_01",
            "mouse_02",
            "mouse_03",
        ]

3. After dataset loading: use_group_split = groups is not None

4. Replace: 

            rskf = RepeatedStratifiedKFold(
                n_splits=5,
                n_repeats=5,
                random_state=seed,
            )

            for fold, (train_idx, test_idx) in enumerate(
                rskf.split(X, y),
                start=1,
            ):

    With this:

            if use_group_split:

                print("Using patient/group-aware splitting.")

                sgkf = StratifiedGroupKFold(
                    n_splits=5,
                    shuffle=True,
                    random_state=seed,
                )

                split_iterator = sgkf.split(
                    X,
                    y,
                    groups=groups,
                )

            else:

                print(
                    "No patient/group metadata found. "
                    "Using image-level stratified splitting."
                )

                rskf = RepeatedStratifiedKFold(
                    n_splits=5,
                    n_repeats=5,
                    random_state=seed,
                )

                split_iterator = rskf.split(X, y)

            for fold, (train_idx, test_idx) in enumerate(
                split_iterator,
                start=1,
            ):

5. Update build_dataset()

    Inside dataset.py:
        parse patient IDs / mouse IDs from metadata
        return groups list alongside X and y

    Example: return X, y, used_files, image_paths, groups

    Remember:
        NEVER use image-level random splitting for patient-based datasets 
        all images from same patient/mouse MUST stay together
        prevents data leakage and inflated metrics
        especially important for histopathology pipelines
'''
