import numpy as np
import pandas as pd
from src.statistics_utils import verify_same_splits

from scipy.stats import (
    t,
    wilcoxon,
    ttest_ind,
    mannwhitneyu
)

CV_N_SPLITS = 5
TEST_FRACTION = 1.0 / CV_N_SPLITS

def load_fold_metric(results_dir, metric = "accuracy"):
    df = pd.read_csv(f"{results_dir}/fold_metrics.csv")
    return df.sort_values("fold")[metric].values

def paired_ttest_corrected(
    a: np.ndarray,
    b: np.ndarray,
    test_fraction: float
    ):
    
    diff = np.asarray(a) - np.asarray(b)

    n = len(diff)

    mean_diff = diff.mean()
    std_diff = diff.std(ddof = 1)

    if np.isclose(std_diff, 0):
        return 0.0, 1.0

    rho = test_fraction

    corrected_se = std_diff * np.sqrt((1.0 / n) + (rho / (1.0 - rho)))

    t_stat = mean_diff / corrected_se

    p_val = 2 * (1 - t.cdf(abs(t_stat), df=n - 1))

    return t_stat, p_val

def independent_tests(a, b):
    t_stat, t_p = ttest_ind(
        a,
        b,
        equal_var = False
    )

    u_stat, u_p = mannwhitneyu(
        a,
        b,
        alternative = "two-sided"
    )

    return (
        t_stat,
        t_p,
        u_stat,
        u_p
    )

def compare_experiments(exp1_dir, exp2_dir, comparison_name, metric = "accuracy", paired = True):
    a = load_fold_metric(exp1_dir, metric)
    b = load_fold_metric(exp2_dir, metric)

    if paired and len(a) != len(b):
        raise ValueError("Experiments must have the same number of folds.")

    if paired:
        same_splits = verify_same_splits(exp1_dir, exp2_dir)

        if not same_splits:
            raise ValueError(
                "Paired comparison requires identical CV splits."
            )
        
        test_fraction = TEST_FRACTION

        t_stat, t_p = paired_ttest_corrected(
            a,
            b,
            test_fraction = test_fraction
        )

        diff = a - b

        if np.allclose(diff, 0):
            w_stat, w_p = 0.0, 1.0
        else:
            w_stat, w_p = wilcoxon(
                a,
                b,
                zero_method = "wilcox",
                alternative = "two-sided"
            )

        test_name_1 = "Corrected paired t-test"
        test_name_2 = "Wilcoxon signed-rank test"

    else:
        t_stat, t_p, w_stat, w_p = independent_tests(a, b)

        diff = a - b

        test_name_1 = "Welch's t-test"
        test_name_2 = "Mann-Whitney U test"

    if paired:
        if np.isclose(diff.std(ddof = 1), 0):
            effect_size = 0.0
        else:
            effect_size = (
                diff.mean() /
                diff.std(ddof = 1)
            )

    else:
        pooled_std = np.sqrt(
            (
                ((len(a) - 1) * a.var(ddof = 1)) +
                ((len(b) - 1) * b.var(ddof = 1))
            ) / (len(a) + len(b) - 2)
        )

        if np.isclose(pooled_std, 0):
            effect_size = 0.0
        else:
            effect_size = (
                (a.mean() - b.mean()) /
                pooled_std
            )

    print(f"\nMetric: {metric}")
    print(f"{exp1_dir}: mean = {a.mean():.4f}, std = {a.std(ddof = 1):.4f}")
    print(f"{exp2_dir}: mean = {b.mean():.4f}, std = {b.std(ddof = 1):.4f}")
    print(
        f"{test_name_1}: "
        f"statistic = {t_stat:.4f}, p = {t_p:.6f}"
    )
    print(
        f"{test_name_2}: "
        f"statistic = {w_stat:.4f}, p = {w_p:.6f}"
    )
    print(f"Effect size (standardized mean difference): {effect_size:.4f}")

    return {
        "comparison": comparison_name,
        "experiment_1": exp1_dir,
        "experiment_2": exp2_dir,
        "metric": metric,
        "test_1_stat": t_stat,
        "test_1_p": t_p,
        "test_2_stat": w_stat,
        "test_2_p": w_p,
        "effect_size": effect_size
    }

if __name__ == "__main__":

    metrics = [
        "accuracy",
        "auc",
        "pr_auc",
        "f1_score",
        "mcc"
    ]
    
    experiment_pairs = [

    # =========================================================
    # MODEL COMPARISONS
    # =========================================================

    (
        "results/seed_42/svm_scaled_all_no_norm_full_100x_seed_42",
        "results/seed_42/rf_unscaled_all_no_norm_full_100x_seed_42",
        "model_comparison_full_100x_no_norm",
        True
    ),

    (
        "results/seed_42/svm_scaled_all_no_norm_full_400x_seed_42",
        "results/seed_42/rf_unscaled_all_no_norm_full_400x_seed_42",
        "model_comparison_full_400x_no_norm",
        True
    ),

    (
        "results/seed_42/svm_scaled_all_reinhard_full_100x_seed_42",
        "results/seed_42/rf_unscaled_all_reinhard_full_100x_seed_42",
        "model_comparison_full_100x_reinhard",
        True
    ),

    (
        "results/seed_42/svm_scaled_all_reinhard_full_400x_seed_42",
        "results/seed_42/rf_unscaled_all_reinhard_full_400x_seed_42",
        "model_comparison_full_400x_reinhard",
        True
    ),

    # =========================================================
    # MAGNIFICATION EFFECTS
    # =========================================================

    (
        "results/seed_42/svm_scaled_all_no_norm_full_100x_seed_42",
        "results/seed_42/svm_scaled_all_no_norm_full_400x_seed_42",
        "magnification_effect_svm_no_norm",
        False
    ),

    (
        "results/seed_42/rf_unscaled_all_no_norm_full_100x_seed_42",
        "results/seed_42/rf_unscaled_all_no_norm_full_400x_seed_42",
        "magnification_effect_rf_no_norm",
        False
    ),

    (
        "results/seed_42/svm_scaled_all_reinhard_full_100x_seed_42",
        "results/seed_42/svm_scaled_all_reinhard_full_400x_seed_42",
        "magnification_effect_svm_reinhard",
        False
    ),

    (
        "results/seed_42/rf_unscaled_all_reinhard_full_100x_seed_42",
        "results/seed_42/rf_unscaled_all_reinhard_full_400x_seed_42",
        "magnification_effect_rf_reinhard",
        False
    ),

    # =========================================================
    # STAIN NORMALIZATION EFFECTS
    # =========================================================

    (
        "results/seed_42/svm_scaled_all_no_norm_full_100x_seed_42",
        "results/seed_42/svm_scaled_all_reinhard_full_100x_seed_42",
        "stain_norm_effect_svm_100x",
        True
    ),

    (
        "results/seed_42/svm_scaled_all_no_norm_full_400x_seed_42",
        "results/seed_42/svm_scaled_all_reinhard_full_400x_seed_42",
        "stain_norm_effect_svm_400x",
        True
    ),

    (
        "results/seed_42/rf_unscaled_all_no_norm_full_100x_seed_42",
        "results/seed_42/rf_unscaled_all_reinhard_full_100x_seed_42",
        "stain_norm_effect_rf_100x",
        True
    ),

    (
        "results/seed_42/rf_unscaled_all_no_norm_full_400x_seed_42",
        "results/seed_42/rf_unscaled_all_reinhard_full_400x_seed_42",
        "stain_norm_effect_rf_400x",
        True
    ),

    # =========================================================
    # FEATURE SET EFFECTS
    # =========================================================

    (
        "results/seed_42/svm_scaled_haralick_no_norm_full_100x_seed_42",
        "results/seed_42/svm_scaled_all_no_norm_full_100x_seed_42",
        "haralick_vs_all_svm_100x",
        True
    ),

    (
        "results/seed_42/rf_unscaled_haralick_no_norm_full_100x_seed_42",
        "results/seed_42/rf_unscaled_all_no_norm_full_100x_seed_42",
        "haralick_vs_all_rf_100x",
        True
    ),

    (
        "results/seed_42/svm_scaled_lbp_no_norm_full_100x_seed_42",
        "results/seed_42/svm_scaled_all_no_norm_full_100x_seed_42",
        "lbp_vs_all_svm_100x",
        True
    ),

    (
        "results/seed_42/rf_unscaled_lbp_no_norm_full_100x_seed_42",
        "results/seed_42/rf_unscaled_all_no_norm_full_100x_seed_42",
        "lbp_vs_all_rf_100x",
        True
    ),

    (
        "results/seed_42/svm_scaled_color_no_norm_full_100x_seed_42",
        "results/seed_42/svm_scaled_all_no_norm_full_100x_seed_42",
        "color_vs_all_svm_100x",
        True
    ),

    (
        "results/seed_42/rf_unscaled_color_no_norm_full_100x_seed_42",
        "results/seed_42/rf_unscaled_all_no_norm_full_100x_seed_42",
        "color_vs_all_rf_100x",
        True
    ),

    # =========================================================
    # SCALING EFFECTS
    # =========================================================

    (
        "results/seed_42/svm_scaled_all_no_norm_full_100x_seed_42",
        "results/seed_42/svm_unscaled_all_no_norm_full_100x_seed_42",
        "scaling_effect_svm_100x",
        True
    ),

    (
        "results/seed_42/svm_scaled_all_no_norm_full_400x_seed_42",
        "results/seed_42/svm_unscaled_all_no_norm_full_400x_seed_42",
        "scaling_effect_svm_400x",
        True
    ),

    # =========================================================
    # DATASET SIZE EFFECTS
    # =========================================================

    (
        "results/seed_42/svm_scaled_all_no_norm_20img_100x_seed_42",
        "results/seed_42/svm_scaled_all_no_norm_full_100x_seed_42",
        "dataset_size_effect_svm_20_vs_full_100x",
        False
    ),

    (
        "results/seed_42/rf_unscaled_all_no_norm_20img_100x_seed_42",
        "results/seed_42/rf_unscaled_all_no_norm_full_100x_seed_42",
        "dataset_size_effect_rf_20_vs_full_100x",
        False
    ),

    (
        "results/seed_42/svm_scaled_all_no_norm_89img_100x_seed_42",
        "results/seed_42/svm_scaled_all_no_norm_full_100x_seed_42",
        "dataset_size_effect_svm_89_vs_full_100x",
        False
    ),

    (
        "results/seed_42/rf_unscaled_all_no_norm_89img_100x_seed_42",
        "results/seed_42/rf_unscaled_all_no_norm_full_100x_seed_42",
        "dataset_size_effect_rf_89_vs_full_100x",
        False
    )
]

    results = []

    for exp1, exp2, comparison_name, paired in experiment_pairs:
        for metric in metrics:

                result = compare_experiments(
                    exp1,
                    exp2,
                    comparison_name,
                    metric = metric,
                    paired = paired
                )

                results.append(result)

    results_df = pd.DataFrame(results)

    n_tests = len(results_df)

    results_df["test_1_p_bonferroni"] = np.minimum(
        results_df["test_1_p"] * n_tests,
        1.0
    )

    results_df["test_2_p_bonferroni"] = np.minimum(
        results_df["test_2_p"] * n_tests,
        1.0
    )

    results_df.to_csv(
        "statistical_tests.csv",
        index=False
    )
