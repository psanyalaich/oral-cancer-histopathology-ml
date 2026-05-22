from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def get_model(model_type, use_scaling=True):

    if model_type == "rf":

        model = RandomForestClassifier(
            class_weight="balanced",
            random_state=42,
        )

        pipeline = Pipeline([
            ("classifier", model),
        ])

    elif model_type == "svm":

        model = SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=42,
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

        return {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [None, 10, 20],
            "classifier__min_samples_split": [2, 5],
            "classifier__min_samples_leaf": [1, 2],
        }

    elif model_type == "svm":

        return {
            "classifier__C": [0.1, 1, 10],
            "classifier__gamma": ["scale", 0.01, 0.001],
        }

    else:
        raise ValueError(f"Unknown model type: {model_type}")
