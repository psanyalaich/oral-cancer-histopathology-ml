import os
import csv
import numpy as np

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

from dataset import build_dataset
from src.features import get_feature_names
from visualize_results import plot_roc_curve, plot_feature_importance

from experiments import EXPERIMENTS


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

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", model),
    ])

    return pipeline


SUMMARY_CSV = "results_summary.csv"

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
            "accuracy",
            "auc",
            "sensitivity",
            "specificity",
        ])


# experiment configurations
for config in EXPERIMENTS:

    try:

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

        # build dataset
        X, y = build_dataset(
            magnification=MAGNIFICATION,
            num_normal=NUM_NORMAL,
            num_tumour=NUM_TUMOUR,
            use_haralick=USE_HARALICK,
        )

        print("Dataset Shape:", X.shape)

        # cross validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        accuracies = []
        aucs = []
        sensitivities = []
        specificities = []

        all_y_test = []
        all_y_prob = []

        # training loop

        print(f"Model: {MODEL_TYPE}")
        print(f"Magnification: {MAGNIFICATION}")
        print(f"Haralick: {USE_HARALICK}")
        print(f"Normal: {NUM_NORMAL}")
        print(f"Tumour: {NUM_TUMOUR}")

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
            print(f"\nFold {fold}")

            X_train = X[train_idx]
            X_test = X[test_idx]

            y_train = y[train_idx]
            y_test = y[test_idx]


            clf = get_model(MODEL_TYPE)
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)[:, 1]

            all_y_test.extend(y_test)
            all_y_prob.extend(y_prob)

            accuracy = accuracy_score(y_test, y_pred)
            auc_score = roc_auc_score(y_test, y_prob)

            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            sensitivity = tp / (tp + fn + 1e-6)
            specificity = tn / (tn + fp + 1e-6)

            accuracies.append(accuracy)
            aucs.append(auc_score)
            sensitivities.append(sensitivity)
            specificities.append(specificity)

            print("Accuracy:", round(accuracy, 3))
            print("AUC:", round(auc_score, 3))
            print("Sensitivity:", round(sensitivity, 3))
            print("Specificity:", round(specificity, 3))

        # save metrics
        metrics_path = os.path.join(RESULTS_DIR, "metrics.txt")

        with open(metrics_path, "w") as f:
            f.write(f"Experiment: {EXPERIMENT_NAME}\n")
            f.write(f"Magnification: {MAGNIFICATION}\n")
            f.write(f"Model: {MODEL_TYPE}\n")
            f.write(f"Use Haralick: {USE_HARALICK}\n")
            f.write(f"Normal images: {NUM_NORMAL}\n")
            f.write(f"Tumour images: {NUM_TUMOUR}\n")
            f.write("\n")
            f.write(f"Mean Accuracy: {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}\n")
            f.write(f"Mean AUC: {np.mean(aucs):.3f}\n")
            f.write(f"Mean Sensitivity: {np.mean(sensitivities):.3f}\n")
            f.write(f"Mean Specificity: {np.mean(specificities):.3f}\n")

        print("FINAL RESULTS")
        print("Mean Accuracy:", round(np.mean(accuracies), 3))
        print("Mean AUC:", round(np.mean(aucs), 3))
        print("Mean Sensitivity:", round(np.mean(sensitivities), 3))
        print("Mean Specificity:", round(np.mean(specificities), 3))


        with open(SUMMARY_CSV, "a", newline="") as f:

            writer = csv.writer(f)

            writer.writerow([
                EXPERIMENT_NAME,
                MODEL_TYPE,
                MAGNIFICATION,
                USE_HARALICK,
                NUM_NORMAL,
                NUM_TUMOUR,
                round(np.mean(accuracies), 3),
                round(np.mean(aucs), 3),
                round(np.mean(sensitivities), 3),
                round(np.mean(specificities), 3),
            ])


        # save ROC
        plot_roc_curve(
            all_y_test,
            all_y_prob,
            os.path.join(RESULTS_DIR, "roc_curve.png"),
        )

        # save feature_importance - only for rf model
        if MODEL_TYPE == "rf":
            feature_names = get_feature_names(USE_HARALICK)

            final_rf = get_model("rf")
            final_rf.fit(X, y) # RF part of pipeline so can't be alone

            rf_model = final_rf.named_steps["classifier"]

            plot_feature_importance(
                feature_names,
                rf_model.feature_importances_,
                os.path.join(RESULTS_DIR, "feature_importance.png"),
            )
        
    except Exception as e:
        print(f"Experiment failed: {EXPERIMENT_NAME}")
        print(e)
