# imports
from itertools import product

# configurations
MODELS = ["rf", "svm"]
MAGNIFICATIONS = ["100x", "400x",]

FEATURE_SETS = [
    "color",
    "lbp",
    "haralick",
    "color_lbp",
    "color_haralick",
    "lbp_haralick",
    "all",
]

DATASET_CONFIGS = {
    "100x": [
        ("20img", 20, 20),
        ("89img", 89, 89),
        ("full", 89, 439),
    ],

    "400x": [
        ("20img", 20, 20),
        ("201img", 201, 201),
        ("full", 201, 495),
    ],
}

# experiment generator
def generate_experiment_name(
    model,
    feature_set,
    dataset_name,
    magnification,
    use_scaling,
    ):
    
    parts = [
        model,
        "scaled" if use_scaling else "unscaled",
    ]

    parts.append(feature_set)

    parts.extend([
        dataset_name,
        magnification
    ])

    return "_".join(parts)

def generate_experiments():
    experiments = []

    for model, magnification, feature_set in product(
        MODELS,
        MAGNIFICATIONS,
        FEATURE_SETS,
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
                    ),

                    "model": model,
                    "feature_set": feature_set,
                    "magnification": magnification,
                    "num_normal": num_normal,
                    "num_tumour": num_tumour,
                    "use_scaling": use_scaling,
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
    }

    names = set()

    for exp in experiments:
        # missing keys
        missing = required_keys - exp.keys()
        assert not missing, \
            f"Missing keys: {missing}"
        # duplicate names
        assert exp["name"] not in names, \
            f"Duplicate experiment name: {exp['name']}"
        names.add(exp["name"])
        # basic checks
        assert exp["num_normal"] > 0
        assert exp["num_tumour"] > 0

    print("All experiments validated!")

# final experiment list
EXPERIMENTS = generate_experiments()
validate_experiments(EXPERIMENTS)
