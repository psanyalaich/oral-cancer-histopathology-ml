import os
import numpy as np

from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
)

from src.model_utils import (
    get_model,
    get_param_grid,
)

from src.metrics_utils import compute_all_metrics
from src.visualize_results import plot_confusion_matrix
from src.explainability import plot_permutation_importance

def run_fold(
    fold,
    X,
    y,
    image_paths,
    split_dir,
    model_type,
    results_dir,
    experiment_name,
    feature_names,
    use_scaling,
):

    train_idx = np.load(
        os.path.join(
            split_dir,
            f"train_idx_fold_{fold}.npy"
        )
    )

    test_idx = np.load(
        os.path.join(
            split_dir,
            f"test_idx_fold_{fold}.npy"
        )
    )

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    base_model = get_model(
        model_type,
        use_scaling=use_scaling,
    )

    param_grid = get_param_grid(model_type)

    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=StratifiedKFold(
            n_splits=3,
            shuffle=True,
            random_state=42,
        ),
        n_jobs=-1,
    )

    grid.fit(X_train, y_train)

    clf = grid.best_estimator_

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    fold_metrics = compute_all_metrics(
        y_test,
        y_pred,
        y_prob,
    )

    fold_row = {
        "experiment": experiment_name,
        "fold": fold,
        **fold_metrics,
    }

    prediction_rows = []

    test_paths = [image_paths[i] for i in test_idx]

    for yt, yp, ypb, img_path in zip(
        y_test,
        y_pred,
        y_prob,
        test_paths,
    ):

        prediction_rows.append({
            "experiment": experiment_name,
            "model": model_type,
            "fold": fold,
            "image_path": img_path,
            "y_true": int(yt),
            "y_pred": int(yp),
            "y_prob": float(ypb),
        })

    if model_type == "rf":

        plot_permutation_importance(
            clf,
            X_test,
            y_test,
            feature_names,
            os.path.join(
                results_dir,
                f"permutation_importance_fold_{fold}.png",
            ),
        )

    plot_confusion_matrix(
        y_test,
        y_pred,
        os.path.join(
            results_dir,
            f"confusion_matrix_fold_{fold}.png"
        ),
        normalize=True,
    )

    return {
        "fold_row": fold_row,
        "prediction_rows": prediction_rows,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "best_params": {
            "fold": fold,
            "best_cv_auc": grid.best_score_,
            **grid.best_params_,
        },
    }
