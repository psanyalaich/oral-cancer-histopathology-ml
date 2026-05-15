import matplotlib
matplotlib.use("Agg")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, 
    precision_recall_curve, 
    confusion_matrix, 
    ConfusionMatrixDisplay,
    average_precision_score
)

def plot_roc_curve(y_true, y_prob, save_path, csv_path=None):
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    if csv_path is not None:
        roc_df = pd.DataFrame({
            "fpr": fpr,
            "tpr": tpr,
            "threshold": thresholds,
        })

        roc_df.to_csv(
            csv_path, 
            index=False,
        )

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_pr_curve(y_true, y_prob, save_path, csv_path=None):
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    # Average Precision (recommended for imbalanced datasets)
    pr_auc = average_precision_score(y_true, y_prob)

    if csv_path is not None:

        # thresholds array is shorter by 1 element
        threshold_array = np.append(
            thresholds,
            np.nan,
        )

        pr_df = pd.DataFrame({
            "precision": precision,
            "recall": recall,
            "threshold": threshold_array,
        })

        pr_df.to_csv(
            csv_path,
            index=False,
        )
        
    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, label=f"PR-AUC = {pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, save_path, normalize=True):
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Tumour"])
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", values_format=".2f" if normalize else "d", colorbar=True)
    plt.title("Normalized Confusion Matrix" if normalize else "Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_feature_importance(feature_names, importances, save_path):
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(12, 5))
    plt.bar(feature_names, importances)
    plt.xticks(rotation=45)
    plt.title("Feature Importances")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
