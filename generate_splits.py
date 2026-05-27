import os
import numpy as np
from src.dataset import build_dataset
from experiments import EXPERIMENTS
from sklearn.model_selection import RepeatedStratifiedKFold
from configs.experiment_config import (
    N_SPLITS,
    N_REPEATS
)

def generate_splits():
    generated = set()

    for config in EXPERIMENTS:
        magnification = config["magnification"]
        num_normal = config["num_normal"]
        num_tumour = config["num_tumour"]
        stain_normalization = config["stain_normalization"]
        seed = config["seed"]

        split_key = (
            magnification,
            num_normal,
            num_tumour,
            seed
        )

        if split_key in generated:
            continue

        generated.add(split_key)

        print("\nGenerating splits for:")
        print(split_key)

        X, y, used_files, image_paths = build_dataset(
            magnification = magnification,
            num_normal = num_normal,
            num_tumour = num_tumour,
            feature_set = config["feature_set"],
            stain_normalization = stain_normalization
        )

        split_dir = (
            f"cv_splits/"
            f"{magnification}_"
            f"{num_normal}_"
            f"{num_tumour}_"
            f"seed_{seed}"
        )

        os.makedirs(split_dir, exist_ok = True)

        rskf = RepeatedStratifiedKFold(
            n_splits = N_SPLITS,
            n_repeats = N_REPEATS,
            random_state = seed
        )

        split_iterator = enumerate(
            rskf.split(X, y),
            start=0
        )

        for iteration_idx, (train_idx, test_idx) in split_iterator:

            repeat = (iteration_idx // N_SPLITS) + 1
            fold = (iteration_idx % N_SPLITS) + 1

            np.save(
                os.path.join(
                    split_dir,
                    f"train_idx_repeat_{repeat}_fold_{fold}.npy"
                ),
                train_idx
            )

            np.save(
                os.path.join(
                    split_dir,
                    f"test_idx_repeat_{repeat}_fold_{fold}.npy"
                ),
                test_idx
            )

    print("\nAll splits generated.")


if __name__ == "__main__":
    generate_splits()
