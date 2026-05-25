import os
import numpy as np
from src.model_utils import get_model
from src.explainability import plot_shap_summary_rf
from src.visualize_results import plot_feature_importance

def run_rf_analysis(
    X,
    y,
    feature_names,
    results_dir
):

    final_rf = get_model(
        "rf",
        use_scaling=False
    )

    final_rf.fit(X, y)

    rf_model = final_rf.named_steps["classifier"]

    plot_feature_importance(
        feature_names,
        rf_model.feature_importances_,
        os.path.join(
            results_dir,
            "feature_importance.png"
        )
    )

    rng = np.random.default_rng(42)

    sample_idx = rng.choice(
        len(X),
        size=min(200, len(X)),
        replace = False
    )

    X_shap = X[sample_idx]

    plot_shap_summary_rf(
        final_rf,
        X_shap,
        feature_names,
        os.path.join(
            results_dir,
            "shap_summary.png"
        )
    )
