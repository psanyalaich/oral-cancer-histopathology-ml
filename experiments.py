from itertools import product

# configurations
MODELS = ["rf", "svm"]
MAGNIFICATIONS = ["100x", "400x",]
HARALICK_OPTIONS = [False, True,]
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
    haralick,
    dataset_name,
    magnification,
    ):
    
    parts = [model]

    if haralick:
        parts.append("haralick")

    parts.extend([
        dataset_name,
        magnification
    ])

    return "_".join(parts)

def generate_experiments():
    experiments = []

    for model, magnification, haralick in product(
        MODELS,
        MAGNIFICATIONS,
        HARALICK_OPTIONS,
    ):

        datasets = DATASET_CONFIGS[magnification]

        for dataset_name, num_normal, num_tumour in datasets:
            experiment = {
                "name": generate_experiment_name(
                    model,
                    haralick,
                    dataset_name,
                    magnification,
                ),

                "model": model,
                "haralick": haralick,
                "magnification": magnification,
                "num_normal": num_normal,
                "num_tumour": num_tumour,
            }

            experiments.append(experiment)
    return experiments

def validate_experiments(experiments):
    required_keys = {
        "name",
        "model",
        "haralick",
        "magnification",
        "num_normal",
        "num_tumour",
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
