# IMPORTS
import os
import json
import logging
import matplotlib
import pandas as pd
matplotlib.use("Agg")

from src.features import get_feature_names
from src.experiment_runner import run_fold
from src.rf_analysis import run_rf_analysis
from src.evaluation_utils import save_overall_evaluation

from src.results_utils import (
    initialize_summary_csv,
    save_fold_metrics,
    save_predictions,
    save_best_params,
    write_metrics_txt,
    append_summary_row,
)

from src.analysis_plots import (
    plot_class_distribution, 
    plot_feature_correlation
)

from src.cache_utils import get_or_cache_dataset

from experiments import EXPERIMENTS

N_FOLDS = 25

def run_experiment(config):

    EXPERIMENT_NAME = config["name"]
    MODEL_TYPE = config["model"]
    FEATURE_SET = config["feature_set"]
    MAGNIFICATION = config["magnification"]
    NUM_NORMAL = config["num_normal"]
    NUM_TUMOUR = config["num_tumour"]
    USE_SCALING = config["use_scaling"]

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
        feature_set=FEATURE_SET,
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

    feature_names = get_feature_names(FEATURE_SET)

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
                feature_names=feature_names,
                use_scaling=USE_SCALING,
            )

            fold_rows.append(result["fold_row"])

            prediction_rows.extend(result["prediction_rows"])

            all_y_test.extend(result["y_test"])
            all_y_pred.extend(result["y_pred"])
            all_y_prob.extend(result["y_prob"])

            best_params_rows.append(result["best_params"])

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

    save_best_params(
        best_params_rows,
        RESULTS_DIR,
    )

    save_fold_metrics(
        fold_df,
        RESULTS_DIR,
    )

    save_predictions(
        prediction_rows,
        RESULTS_DIR,
    )

    write_metrics_txt(
        os.path.join(
            RESULTS_DIR,
            "metrics.txt",
        ),
        config,
        fold_df,
    )

    append_summary_row(
        SUMMARY_CSV,
        config,
        fold_df,
    )

    save_overall_evaluation(
        all_y_test=all_y_test,
        all_y_pred=all_y_pred,
        all_y_prob=all_y_prob,
        results_dir=RESULTS_DIR,
    )

    # RF ANALYSIS
    if MODEL_TYPE == "rf":

        run_rf_analysis(
            X=X,
            y=y,
            feature_names=feature_names,
            results_dir=RESULTS_DIR,
        )

# SUMMARY CSV
SUMMARY_CSV = "results_summary.csv"

initialize_summary_csv(SUMMARY_CSV)

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
