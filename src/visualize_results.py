import os
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve, auc, 
    precision_recall_curve, 
    confusion_matrix, 
    ConfusionMatrixDisplay,
    average_precision_score
)


def ensure_save_dir(save_path):
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok = True)

def plot_roc_curve(
        y_true, 
        y_prob, 
        save_path,
        csv_path = None,
        title="ROC Curve"
        ):
    
    ensure_save_dir(save_path)

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    if csv_path is not None:
        roc_df = pd.DataFrame({
            "fpr": fpr,
            "tpr": tpr,
            "threshold": thresholds
        })

        roc_df.to_csv(
            csv_path, 
            index = False
        )

    plt.figure(figsize = (6, 6))
    plt.plot(fpr, tpr, label = f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi = 300, bbox_inches = "tight")
    plt.close()

def plot_pr_curve(y_true, y_prob, save_path, csv_path = None, title = "Precision-Recall Curve"):
    ensure_save_dir(save_path)

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    pr_auc = average_precision_score(y_true, y_prob)

    if csv_path is not None:
        threshold_array = np.append(
            thresholds,
            np.nan
        )

        pr_df = pd.DataFrame({
            "precision": precision,
            "sensitivity": recall,
            "threshold": threshold_array
        })

        pr_df.to_csv(
            csv_path,
            index = False
        )
        
    plt.figure(figsize = (6, 6))
    plt.plot(recall, precision, label = f"PR-AUC = {pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi = 300, bbox_inches = "tight")
    plt.close()

def plot_calibration_curve(y_true, y_prob, save_path, n_bins=5, csv_path=None, title="Calibration Curve"):
    ensure_save_dir(save_path)

    prob_true, prob_pred = calibration_curve(
        y_true,
        y_prob,
        n_bins = n_bins,
        strategy = "uniform"
    )

    if csv_path is not None:

        calib_df = pd.DataFrame({
            "mean_predicted_probability": prob_pred,
            "fraction_of_positives": prob_true
        })

        calib_df.to_csv(
            csv_path,
            index=False
        )

    plt.figure(figsize = (6, 6))

    plt.plot(
        prob_pred,
        prob_true,
        marker = "o",
        label = "Model"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label = "Perfect Calibration"
    )

    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title(title)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

def plot_confusion_matrix(y_true, y_pred, save_path, normalize = True, csv_path = None, title = None):
    ensure_save_dir(save_path)

    cm = confusion_matrix(y_true, y_pred, labels = [0, 1])

    if csv_path is not None:
        cm_df = pd.DataFrame(
            cm,
            index = ["Normal", "Tumour"],
            columns = ["Pred_Normal", "Pred_Tumour"]
        )

        cm_df.to_csv(csv_path)

    if normalize:
        cm = cm.astype(float) / cm.sum(axis = 1, keepdims = True).clip(min = 1)

    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ["Normal", "Tumour"])
    fig, ax = plt.subplots(figsize = (6, 6))
    disp.plot(ax = ax, cmap = "Blues", values_format = ".2f" if normalize else "d", colorbar = True)
    if title is None:
        title = (
            "Normalized Confusion Matrix"
            if normalize
            else "Confusion Matrix"
        )

    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi = 300, bbox_inches = "tight")
    plt.close()

def plot_feature_importance(feature_names, importances, save_path):
    ensure_save_dir(save_path)

    plt.figure(figsize = (12, 5))
    plt.bar(feature_names, importances)
    plt.xticks(rotation = 45)
    plt.title("Feature Importances")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(save_path, dpi = 300, bbox_inches = "tight")
    plt.close()
