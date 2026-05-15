import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

def load_fold_metric(results_dir, metric="accuracy"):
    df = pd.read_csv(f"{results_dir}/fold_metrics.csv")
    return df.sort_values("fold")[metric].values

def compare_experiments(exp1_dir, exp2_dir, metric="accuracy"):
    a = load_fold_metric(exp1_dir, metric)
    b = load_fold_metric(exp2_dir, metric)

    if len(a) != len(b):
        raise ValueError("Experiments must have the same number of folds.")

    if not verify_same_splits(exp1_dir, exp2_dir):
        raise ValueError(
            "Cross-validation splits do not match. "
            "Paired statistical testing is invalid."
        )
    
    t_stat, t_p = ttest_rel(a, b)
    w_stat, w_p = wilcoxon(a, b)

    diff = a - b

    if diff.std(ddof=1) == 0:
        effect_size = 0.0
    else:
        effect_size = diff.mean() / diff.std(ddof=1)

    print(f"\nMetric: {metric}")
    print(f"{exp1_dir}: mean={a.mean():.4f}, std={a.std():.4f}")
    print(f"{exp2_dir}: mean={b.mean():.4f}, std={b.std():.4f}")
    print(f"Paired t-test: statistic={t_stat:.4f}, p={t_p:.6f}")
    print(f"Wilcoxon test: statistic={w_stat:.4f}, p={w_p:.6f}")
    print(f"Effect size (Cohen's d approximation): {effect_size:.4f}")

    return {
        "metric": metric,
        "paired_t_stat": t_stat,
        "paired_t_p": t_p,
        "wilcoxon_stat": w_stat,
        "wilcoxon_p": w_p,
    }

def verify_same_splits(exp1_dir, exp2_dir, n_folds=25):

    for fold in range(1, n_folds + 1):

        exp1_test = np.load(
            os.path.join(
                exp1_dir,
                f"test_idx_fold_{fold}.npy"
            )
        )

        exp2_test = np.load(
            os.path.join(
                exp2_dir,
                f"test_idx_fold_{fold}.npy"
            )
        )

        if not np.array_equal(exp1_test, exp2_test):
            return False

    return True

if __name__ == "__main__":

    metrics=[
        "accuracy",
        "auc",
        "pr_auc",
        "f1_score",
        "mcc",
    ]
    
    experiment_pairs = [
        (
        "results/svm_full_100x", 
        "results/rf_full_100x", 
        ),
        (
        "results/svm_haralick_full_100x", 
        "results/svm_full_100x",
        )
    ]

    # RESULTS
    results = []

    for exp1, exp2 in experiment_pairs:
        for metric in metrics:

                result = compare_experiments(
                    exp1,
                    exp2,
                    metric=metric,
                )

                results.append(result)

    pd.DataFrame(results).to_csv(
        "statistical_tests.csv",
        index=False,
    )
