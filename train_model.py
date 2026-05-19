# IMPORTS
import os
import csv
import json
import logging
import matplotlib
import numpy as np
import pandas as pd
matplotlib.use("Agg")

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from src.features import get_feature_names
from src.metrics_utils import compute_all_metrics
from src.statistics_utils import confidence_interval

from src.analysis_plots import (
    plot_class_distribution, 
    plot_feature_correlation
)

from src.explainability import (
    plot_permutation_importance,
    plot_shap_summary_rf
)

from src.cache_utils import get_or_cache_dataset

from visualize_results import (
    plot_roc_curve, 
    plot_pr_curve, 
    plot_confusion_matrix, 
    plot_feature_importance
)

from experiments import EXPERIMENTS

# MODEL
def get_model(model_type):

    if model_type == "rf":
        model = RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42,
        )

    elif model_type == "svm":
        model = SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=42,
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if model_type == "rf":
        pipeline = Pipeline([
            ("classifier", model),
        ])

    elif model_type == "svm":
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", model),
        ])

    return pipeline

def run_fold(
    fold,
    X,
    y,
    image_paths,
    split_dir,
    model_type,
    results_dir,
    experiment_name,
    ):

    train_idx = np.load(
        os.path.join(
            split_dir,
            f"train_idx_fold_{fold}.npy"
        )
    )

    test_idx = np.load(
        os.path.join(
            split_dir,
            f"test_idx_fold_{fold}.npy"
        )
    )

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    print(f"\nFold {fold}")

    base_model = get_model(model_type)

    # PARAM GRID
    if model_type == "rf":

        param_grid = {
            "classifier__n_estimators": [100],
            "classifier__max_depth": [None, 10],
        }

    elif model_type == "svm":

        param_grid = {
            "classifier__C": [0.1, 1],
            "classifier__gamma": ["scale", 0.01],
        }

    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=StratifiedKFold(
            n_splits=3,
            shuffle=True,
            random_state=42,
        ),
        n_jobs=-1,
    )

    grid.fit(X_train, y_train)

    clf = grid.best_estimator_

    print("Best Params:", grid.best_params_)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    fold_metrics = compute_all_metrics(
        y_test,
        y_pred,
        y_prob,
    )

    fold_row = {
        "experiment": experiment_name,
        "fold": fold,
        **fold_metrics,
    }

    prediction_rows = []

    test_paths = [image_paths[i] for i in test_idx]

    for yt, yp, ypb, img_path in zip(
        y_test,
        y_pred,
        y_prob,
        test_paths,
    ):

        prediction_rows.append({
            "experiment": experiment_name,
            "model": model_type,
            "fold": fold,
            "image_path": img_path,
            "y_true": int(yt),
            "y_pred": int(yp),
            "y_prob": float(ypb),
        })

    print("Accuracy:", round(fold_metrics["accuracy"], 3))
    print("AUC:", round(fold_metrics["auc"], 3))

    plot_confusion_matrix(
        y_test,
        y_pred,
        os.path.join(
            results_dir,
            f"confusion_matrix_fold_{fold}.png"
        ),
        normalize=True,
    )

    return {
        "fold_row": fold_row,
        "prediction_rows": prediction_rows,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "best_params": {
            "fold": fold,
            "best_cv_auc": grid.best_score_,
            **grid.best_params_,
        },
        "last_X_test": X_test,
        "last_y_test": y_test,
    }

def run_experiment(config):

    EXPERIMENT_NAME = config["name"]
    MODEL_TYPE = config["model"]
    USE_HARALICK = config["haralick"]
    MAGNIFICATION = config["magnification"]
    NUM_NORMAL = config["num_normal"]
    NUM_TUMOUR = config["num_tumour"]

    print("\n===================================")
    print("RUNNING:", EXPERIMENT_NAME)
    print("===================================")

    RESULTS_DIR = os.path.join("results", EXPERIMENT_NAME)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(
        os.path.join(RESULTS_DIR, "config.json"),
        "w"
    ) as f:

        json.dump(config, f, indent=4)

    X, y, used_files, image_paths = get_or_cache_dataset(
        magnification=MAGNIFICATION,
        num_normal=NUM_NORMAL,
        num_tumour=NUM_TUMOUR,
        use_haralick=USE_HARALICK,
    )

    with open(
        os.path.join(
            RESULTS_DIR,
            "dataset_manifest.json",
        ),
        "w",
    ) as f:

        json.dump(
            used_files,
            f,
            indent=4,
        )

    feature_names = get_feature_names(USE_HARALICK)

    plot_class_distribution(
        y,
        os.path.join(
            RESULTS_DIR,
            "class_distribution.png",
        ),
        title=f"Class Distribution - {EXPERIMENT_NAME}"
    )

    plot_feature_correlation(
        X,
        feature_names,
        os.path.join(
            RESULTS_DIR,
            "feature_correlation.png",
        ),
        title=f"Feature Correlation - {EXPERIMENT_NAME}"
    )

    fold_rows = []
    prediction_rows = []

    all_y_test = []
    all_y_pred = []
    all_y_prob = []

    best_params_rows = []

    SPLIT_DIR = (
        f"cv_splits/"
        f"{MAGNIFICATION}_"
        f"{NUM_NORMAL}_"
        f"{NUM_TUMOUR}"
    )

    N_FOLDS = 25

    for fold in range(1, N_FOLDS + 1):

        try:

            result = run_fold(
                fold=fold,
                X=X,
                y=y,
                image_paths=image_paths,
                split_dir=SPLIT_DIR,
                model_type=MODEL_TYPE,
                results_dir=RESULTS_DIR,
                experiment_name=EXPERIMENT_NAME,
            )

            fold_rows.append(result["fold_row"])

            prediction_rows.extend(result["prediction_rows"])

            all_y_test.extend(result["y_test"])
            all_y_pred.extend(result["y_pred"])
            all_y_prob.extend(result["y_prob"])

            best_params_rows.append(result["best_params"])

            last_X_test = result["last_X_test"]
            last_y_test = result["last_y_test"]

        except Exception:

            logging.exception(
                f"Fold {fold} failed in {EXPERIMENT_NAME}"
            )

            raise

    fold_df = pd.DataFrame(fold_rows)

    print("\nFINAL RESULTS")

    for metric in [
        "accuracy",
        "auc",
        "pr_auc",
        "precision",
        "sensitivity",
        "f1_score",
        "specificity",
        "mcc",
    ]:

        print(
            f"{metric}: "
            f"{fold_df[metric].mean():.3f} ± "
            f"{fold_df[metric].std():.3f}"
        )

    # SAVE BEST PARAMS
    best_params_df = pd.DataFrame(best_params_rows)

    best_params_df.to_csv(
        os.path.join(
            RESULTS_DIR,
            "best_hyperparameters.csv",
        ),
        index=False,
    )

    # SAVE FOLD RESULTS
    fold_df.to_csv(
        os.path.join(
            RESULTS_DIR,
            "fold_metrics.csv",
        ),
        index=False,
    )

    # SAVE PREDICTIONS
    pred_df = pd.DataFrame(prediction_rows)

    pred_df.to_csv(
        os.path.join(
            RESULTS_DIR,
            "fold_predictions.csv",
        ),
        index=False,
    )

    # SUMMARY METRICS
    summary = fold_df.drop(
        columns=["experiment", "fold"]
    ).agg(["mean", "std"]).T

    summary.to_csv(
        os.path.join(
            RESULTS_DIR,
            "summary_metrics.csv",
        )
    )

    # METRICS TXT
    metrics_path = os.path.join(
        RESULTS_DIR,
        "metrics.txt",
    )

    with open(metrics_path, "w") as f:

        f.write(f"Experiment: {EXPERIMENT_NAME}\n")
        f.write(f"Magnification: {MAGNIFICATION}\n")
        f.write(f"Model: {MODEL_TYPE}\n")
        f.write(f"Use Haralick: {USE_HARALICK}\n")
        f.write(f"Normal images: {NUM_NORMAL}\n")
        f.write(f"Tumour images: {NUM_TUMOUR}\n\n")

        for metric in [
            "accuracy",
            "auc",
            "pr_auc",
            "precision",
            "f1_score",
            "sensitivity",
            "specificity",
            "mcc",
        ]:

            mean_value = fold_df[metric].mean()
            std_value = fold_df[metric].std()

            mean, ci_low, ci_high = confidence_interval(
                fold_df[metric].values
            )

            f.write(
                f"{metric}: "
                f"{mean_value:.3f} ± {std_value:.3f}\n"
                f"(95% CI: {ci_low:.3f} - {ci_high:.3f})\n"
            )

    # APPEND TO SUMMARY CSV
    with open(SUMMARY_CSV, "a", newline="") as f:

        writer = csv.writer(f)

        accuracy_mean, accuracy_ci_low, accuracy_ci_high = confidence_interval(
            fold_df["accuracy"].values
        )

        writer.writerow([

            EXPERIMENT_NAME,
            MODEL_TYPE,
            MAGNIFICATION,
            USE_HARALICK,
            NUM_NORMAL,
            NUM_TUMOUR,

            round(fold_df["accuracy"].mean(), 3),
            round(fold_df["accuracy"].std(), 3),

            round(fold_df["auc"].mean(), 3),
            round(fold_df["auc"].std(), 3),

            round(fold_df["pr_auc"].mean(), 3),
            round(fold_df["pr_auc"].std(), 3),

            round(fold_df["precision"].mean(), 3),
            round(fold_df["precision"].std(), 3),

            round(fold_df["sensitivity"].mean(), 3),
            round(fold_df["sensitivity"].std(), 3),

            round(fold_df["f1_score"].mean(), 3),
            round(fold_df["f1_score"].std(), 3),

            round(fold_df["specificity"].mean(), 3),
            round(fold_df["specificity"].std(), 3),

            round(fold_df["mcc"].mean(), 3),
            round(fold_df["mcc"].std(), 3),

            round(accuracy_ci_low, 3),
            round(accuracy_ci_high, 3),
        ])

    pooled_auc = roc_auc_score(
        all_y_test,
        all_y_prob,
    )

    # ROC CURVE
    plot_roc_curve(
        all_y_test,
        all_y_prob,
        os.path.join(RESULTS_DIR, "roc_curve.png"),
        pooled_auc=pooled_auc,
        csv_path=os.path.join(
            RESULTS_DIR,
            "roc_curve_data.csv",
        ),
    )

    # PR CURVE
    plot_pr_curve(
        all_y_test,
        all_y_prob,
        os.path.join(RESULTS_DIR, "pr_curve.png"),
        csv_path=os.path.join(
            RESULTS_DIR,
            "pr_curve_data.csv",
        ),
    )

    # OVERALL CONFUSION MATRIX
    plot_confusion_matrix(
        all_y_test,
        all_y_pred,
        os.path.join(
            RESULTS_DIR,
            "confusion_matrix_overall.png",
        ),
        normalize=True,
    )

    cm = confusion_matrix(
        all_y_test,
        all_y_pred,
    )

    cm_df = pd.DataFrame(
        cm,
        index=["Normal", "Tumour"],
        columns=["Pred_Normal", "Pred_Tumour"],
    )

    cm_df.to_csv(
        os.path.join(
            RESULTS_DIR,
            "confusion_matrix_overall.csv",
        )
    )

    # RF ANALYSIS
    if MODEL_TYPE == "rf":

        final_rf = get_model("rf")
        final_rf.fit(X, y)

        rf_model = final_rf.named_steps["classifier"]

        plot_feature_importance(
            feature_names,
            rf_model.feature_importances_,
            os.path.join(
                RESULTS_DIR,
                "feature_importance.png",
            ),
        )

        rng = np.random.default_rng(42)

        sample_idx = rng.choice(
            len(X),
            size=min(200, len(X)),
            replace=False,
        )

        X_shap = X[sample_idx]

        plot_shap_summary_rf(
            final_rf,
            X_shap,
            feature_names,
            os.path.join(
                RESULTS_DIR,
                "shap_summary.png",
            ),
        )

        plot_permutation_importance(
            final_rf,
            last_X_test,
            last_y_test,
            feature_names,
            os.path.join(
                RESULTS_DIR,
                "permutation_importance.png",
            ),
        )

# SUMMARY CSV
SUMMARY_CSV = "results_summary.csv"

if os.path.exists(SUMMARY_CSV):
    os.remove(SUMMARY_CSV)

if not os.path.exists(SUMMARY_CSV):
    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "experiment",
            "model",
            "magnification",
            "haralick",
            "num_normal",
            "num_tumour",

            "accuracy_mean",
            "accuracy_std",

            "auc_mean",
            "auc_std",

            "pr_auc_mean",
            "pr_auc_std",

            "precision_mean",
            "precision_std",

            "f1_score_mean",
            "f1_score_std",

            "sensitivity_mean",
            "sensitivity_std",

            "specificity_mean",
            "specificity_std",

            "mcc_mean",
            "mcc_std",

            "accuracy_ci_low",
            "accuracy_ci_high",
        ])

def main():

    failed_experiments = []

    for config in EXPERIMENTS:

        try:
            run_experiment(config)

        except Exception:

            failed_experiments.append(config["name"])

            logging.exception(
                f"Experiment failed: {config['name']}"
            )

    if failed_experiments:

        print("\nFailed Experiments:")
        for name in failed_experiments:
            print("-", name)

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    main()
