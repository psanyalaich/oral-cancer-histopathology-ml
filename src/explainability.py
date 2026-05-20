import os
import shap
import matplotlib
import numpy as np
import pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

def plot_permutation_importance(
    model,
    X,
    y,
    feature_names,
    save_path,
    n_repeats=10,
):

    save_dir = os.path.dirname(save_path)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=42,
        scoring="roc_auc",
    )

    importances = result.importances_mean
    order = np.argsort(importances)[::-1]

    # SAVE CSV
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    })

    importance_df.sort_values(
        "importance_mean",
        ascending=False,
    ).to_csv(
        save_path.replace(".png", ".csv"),
        index=False,
    )

    plt.figure(figsize=(12, 5))

    plt.bar(
        [feature_names[i] for i in order],
        importances[order],
    )

    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Permutation Importance")
    plt.title("Permutation Feature Importance")

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

def plot_shap_summary_rf(model, X, feature_names, save_path):

    save_dir = os.path.dirname(save_path)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    rf = model.named_steps["classifier"]
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X)

    # binary classification compatibility
    if isinstance(shap_values, list):
        shap_to_plot = shap_values[1]
    elif len(shap_values.shape) == 3:
        shap_to_plot = shap_values[:, :, 1]
    else:
        shap_to_plot = shap_values

    plt.figure()

    shap.summary_plot(
        shap_to_plot,
        X,
        feature_names=feature_names,
        show=False,
    )

    plt.tight_layout()
    
    plt.savefig(
        save_path, 
        dpi=300, 
        bbox_inches="tight",
    )

    plt.close()
