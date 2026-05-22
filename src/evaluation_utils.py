import os

from src.visualize_results import (
    plot_roc_curve,
    plot_pr_curve,
    plot_confusion_matrix,
)

def save_overall_evaluation(
    all_y_test,
    all_y_pred,
    all_y_prob,
    results_dir,
):

    plot_roc_curve(
        all_y_test,
        all_y_prob,
        os.path.join(
            results_dir,
            "roc_curve.png",
        ),
        csv_path=os.path.join(
            results_dir,
            "roc_curve_data.csv",
        ),
    )

    plot_pr_curve(
        all_y_test,
        all_y_prob,
        os.path.join(
            results_dir,
            "pr_curve.png",
        ),
        csv_path=os.path.join(
            results_dir,
            "pr_curve_data.csv",
        ),
    )

    plot_confusion_matrix(
        all_y_test,
        all_y_pred,
        os.path.join(
            results_dir,
            "confusion_matrix_overall.png",
        ),
        normalize=True,
        csv_path=os.path.join(
            results_dir,
            "confusion_matrix_overall.csv",
        ),
    )
