import os
import csv
import pandas as pd

from src.statistics_utils import confidence_interval

def initialize_summary_csv(summary_csv):
    if os.path.exists(summary_csv):
        return

    with open(summary_csv, "w", newline = "") as f:
        writer = csv.writer(f)

        writer.writerow([
            "experiment",
            "model",
            "magnification",
            "feature_set",
            "stain_normalization",
            "num_normal",
            "num_tumour",
            "use_scaling",

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
            "accuracy_ci_high"

        ])

def save_fold_metrics(fold_df, results_dir):
    fold_df.to_csv(
        os.path.join(results_dir, "fold_metrics.csv"),
        index=False
    )

    summary = fold_df.drop(
        columns = ["experiment", "fold"]
    ).agg(["mean", "std"]).T

    summary.to_csv(
        os.path.join(results_dir, "summary_metrics.csv")
    )

def save_predictions(prediction_rows, results_dir):
    pred_df = pd.DataFrame(prediction_rows)

    pred_df.to_csv(
        os.path.join(
            results_dir,
            "fold_predictions.csv"
        ),
        index=False
    )

def save_best_params(best_params_rows, results_dir):
    best_params_df = pd.DataFrame(best_params_rows)

    best_params_df.to_csv(
        os.path.join(
            results_dir,
            "best_hyperparameters.csv"
        ),
        index=False
    )

def write_metrics_txt(
    metrics_path,
    config,
    fold_df
):

    with open(metrics_path, "w") as f:
        f.write(f"Experiment: {config['name']}\n")
        f.write(f"Magnification: {config['magnification']}\n")
        f.write(f"Model: {config['model']}\n")
        f.write(f"Use Feature Set: {config['feature_set']}\n")
        f.write(f"Normal images: {config['num_normal']}\n")
        f.write(f"Tumour images: {config['num_tumour']}\n\n")
        f.write(f"Use Scaling: {config['use_scaling']}\n")

        for metric in [
            "accuracy",
            "auc",
            "pr_auc",
            "precision",
            "f1_score",
            "sensitivity",
            "specificity",
            "mcc"
        ]:

            mean_value = fold_df[metric].mean()
            std_value = fold_df[metric].std()

            _, ci_low, ci_high = confidence_interval(
                fold_df[metric].values
            )

            f.write(
                f"{metric}: "
                f"{mean_value:.3f} ± {std_value:.3f}\n"
                f"(95% CI: {ci_low:.3f} - {ci_high:.3f})\n"
            )

def append_summary_row(
    summary_csv,
    config,
    fold_df
):

    _, accuracy_ci_low, accuracy_ci_high = confidence_interval(
        fold_df["accuracy"].values
    )

    with open(summary_csv, "a", newline = "") as f:

        writer = csv.writer(f)

        writer.writerow([

            config["name"],
            config["model"],
            config["magnification"],
            config["feature_set"],
            config["stain_normalization"],
            config["num_normal"],
            config["num_tumour"],
            config["use_scaling"],

            round(fold_df["accuracy"].mean(), 3),
            round(fold_df["accuracy"].std(), 3),

            round(fold_df["auc"].mean(), 3),
            round(fold_df["auc"].std(), 3),

            round(fold_df["pr_auc"].mean(), 3),
            round(fold_df["pr_auc"].std(), 3),

            round(fold_df["precision"].mean(), 3),
            round(fold_df["precision"].std(), 3),

            round(fold_df["f1_score"].mean(), 3),
            round(fold_df["f1_score"].std(), 3),

            round(fold_df["sensitivity"].mean(), 3),
            round(fold_df["sensitivity"].std(), 3),

            round(fold_df["specificity"].mean(), 3),
            round(fold_df["specificity"].std(), 3),

            round(fold_df["mcc"].mean(), 3),
            round(fold_df["mcc"].std(), 3),

            round(accuracy_ci_low, 3),
            round(accuracy_ci_high, 3)
        ])
