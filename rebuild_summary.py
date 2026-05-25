import os
import pandas as pd

from src.results_utils import (
    initialize_summary_csv,
    append_summary_row
)

SUMMARY_CSV = "results_summary.csv"

if os.path.exists(SUMMARY_CSV):
    os.remove(SUMMARY_CSV)

initialize_summary_csv(SUMMARY_CSV)

results_root = "results"

for seed_folder in os.listdir(results_root):

    seed_path = os.path.join(
        results_root,
        seed_folder
    )

    if not os.path.isdir(seed_path):
        continue

    for experiment_name in os.listdir(seed_path):

        experiment_dir = os.path.join(
            seed_path,
            experiment_name
        )

        if not os.path.isdir(experiment_dir):
            continue

        config_path = os.path.join(
            experiment_dir,
            "config.json"
        )

        fold_metrics_path = os.path.join(
            experiment_dir,
            "fold_metrics.csv"
        )

        if (
            not os.path.exists(config_path)
            or
            not os.path.exists(fold_metrics_path)
        ):
            continue

        config = pd.read_json(
            config_path,
            typ = "series"
        ).to_dict()

        fold_df = pd.read_csv(
            fold_metrics_path
        )

        append_summary_row(
            SUMMARY_CSV,
            config,
            fold_df
        )

        print(f"Added: {experiment_name}")

print("\nSummary CSV rebuilt successfully.")
