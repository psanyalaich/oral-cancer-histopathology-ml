from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from configs.model_config import (
    RF_PARAM_GRID,
    SVM_PARAM_GRID
)

def get_model(model_type, use_scaling = True, seed = 42):
    if model_type == "rf":
        model = RandomForestClassifier(
            class_weight="balanced",
            random_state=seed
        )

        pipeline = Pipeline([
            ("classifier", model)
        ])

    elif model_type == "svm":
        model = SVC(
            kernel = "rbf",
            probability = True,
            class_weight = "balanced",
            random_state = seed
        )

        steps = []

        if use_scaling:
            steps.append(
                ("scaler", StandardScaler())
            )

        steps.append(
            ("classifier", model)
        )

        pipeline = Pipeline(steps)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return pipeline

def get_param_grid(model_type):
    if model_type == "rf":
        return RF_PARAM_GRID
    elif model_type == "svm":
        return SVM_PARAM_GRID
    else:
        raise ValueError(f"Unknown model type: {model_type}")
