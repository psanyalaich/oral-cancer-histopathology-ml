from itertools import product
from configs.dataset_config import DATASET_CONFIGS

from configs.experiment_config import (
    MODELS,
    MAGNIFICATIONS,
    FEATURE_SETS,
    STAIN_NORMALIZATION_OPTIONS,
    SEEDS,
)

def generate_experiment_name(
    model,
    feature_set,
    dataset_name,
    magnification,
    use_scaling,
    stain_normalization,
    seed
    ):
    
    parts = [
        model,
        "scaled" if use_scaling else "unscaled"
    ]

    parts.extend([
        feature_set,
        stain_normalization
            if stain_normalization is not None
            else "no_norm"
    ])

    parts.extend([
        dataset_name,
        magnification,
        f"seed_{seed}"
    ])

    return "_".join(parts)

def generate_experiments():
    experiments = []

    for model, magnification, feature_set, stain_normalization, seed in product(
        MODELS,
        MAGNIFICATIONS,
        FEATURE_SETS,
        STAIN_NORMALIZATION_OPTIONS,
        SEEDS
    ):

        if model == "rf":
            scaling_configs = [False]

        elif model == "svm":
            scaling_configs = [True, False]

        datasets = DATASET_CONFIGS[magnification]

        for use_scaling in scaling_configs:
            for dataset_name, num_normal, num_tumour in datasets:

                experiment = {
                    "name": generate_experiment_name(
                        model,
                        feature_set,
                        dataset_name,
                        magnification,
                        use_scaling,
                        stain_normalization,
                        seed
                    ),

                    "model": model,
                    "feature_set": feature_set,
                    "magnification": magnification,
                    "num_normal": num_normal,
                    "num_tumour": num_tumour,
                    "use_scaling": use_scaling,
                    "stain_normalization": stain_normalization,
                    "seed": seed
                }

                experiments.append(experiment)
    return experiments

def validate_experiments(experiments):
    required_keys = {
        "name",
        "model",
        "feature_set",
        "magnification",
        "num_normal",
        "num_tumour",
        "use_scaling",
        "stain_normalization",
        "seed"
    }

    names = set()

    for exp in experiments:

        missing = required_keys - exp.keys()
        assert not missing, \
            f"Missing keys: {missing}"

        assert exp["name"] not in names, \
            f"Duplicate experiment name: {exp['name']}"
        names.add(exp["name"])

        assert exp["num_normal"] > 0
        assert exp["num_tumour"] > 0

    print("All experiments validated!")

EXPERIMENTS = generate_experiments()
validate_experiments(EXPERIMENTS)
