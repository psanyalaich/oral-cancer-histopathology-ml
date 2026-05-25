import os
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_learning_curve(
    summary_csv,
    save_path,
    metric="auc",
):

    df = pd.read_csv(summary_csv)

    df = df[
        (df["model"] == "rf") &
        (df["feature_set"] == "all")
    ]

    if len(df) == 0:
        raise ValueError(
            "No experiments matched learning curve filter."
        )

    df["total_samples"] = (
        df["num_normal"] +
        df["num_tumour"]
    )

    grouped = df.groupby("total_samples")

    means = grouped[f"{metric}_mean"].mean()
    stds = grouped[f"{metric}_std"].mean()

    x = means.index.values
    y = means.values
    yerr = stds.values

    plt.figure(figsize = (8, 5))

    plt.errorbar(
        x,
        y,
        yerr = yerr,
        marker = "o",
        capsize = 5
    )

    plt.xlabel("Total Number of Samples")
    plt.ylabel(metric.upper())

    plt.title(
        f"Learning Curve ({metric.upper()})"
    )

    plt.grid(True)

    plt.tight_layout()

    save_dir = os.path.dirname(save_path)

    if save_dir:
        os.makedirs(save_dir, exist_ok = True)

    plt.savefig(
        save_path,
        dpi = 300,
        bbox_inches = "tight"
    )

    plt.close()

    print(f"Saved learning curve: {save_path}")
