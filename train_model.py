import os
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

from dataset import build_dataset
from visualize_results import plot_roc_curve, plot_feature_importance
from src.features import get_feature_names

# experiment configurations
EXPERIMENT_NAME = " "
MODEL_TYPE = " " # rf or svm
USE_HARALICK = # True or False
MAGNIFICATION = " "
NUM_NORMAL = 
NUM_TUMOUR = 

RESULTS_DIR = os.path.join("results", EXPERIMENT_NAME)
os.makedirs(RESULTS_DIR, exist_ok=True)

def get_model(model_type):
    if model_type == "rf":
        return RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42,
        )

    if model_type == "svm":
        return SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=42,
        )

    raise ValueError(f"Unknown MODEL_TYPE: {model_type}")

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
for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
    print(f"\nFold {fold}")

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

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
    f.write(f"Mean Accuracy: {np.mean(accuracies):.3f}\n")
    f.write(f"Mean AUC: {np.mean(aucs):.3f}\n")
    f.write(f"Mean Sensitivity: {np.mean(sensitivities):.3f}\n")
    f.write(f"Mean Specificity: {np.mean(specificities):.3f}\n")

print("FINAL RESULTS")
print("Mean Accuracy:", round(np.mean(accuracies), 3))
print("Mean AUC:", round(np.mean(aucs), 3))
print("Mean Sensitivity:", round(np.mean(sensitivities), 3))
print("Mean Specificity:", round(np.mean(specificities), 3))

# save ROC
plot_roc_curve(
    all_y_test,
    all_y_prob,
    os.path.join(RESULTS_DIR, "roc_curve.png"),
)

# save feature_importance - only for rf model
if MODEL_TYPE == "rf":
    feature_names = get_feature_names(USE_HARALICK)

    final_rf = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
    )
    final_rf.fit(X, y)

    plot_feature_importance(
        feature_names,
        final_rf.feature_importances_,
        os.path.join(RESULTS_DIR, "feature_importance.png"),
    )