import os
import numpy as np
from scipy.stats import t

def confidence_interval(values, confidence=0.95):
    values = np.asarray(values)

    n = len(values)

    mean = np.mean(values)
    std = values.std(ddof=1)

    se = std / np.sqrt(n)

    margin = se * t.ppf((1 + confidence) / 2, n - 1)

    return (
        mean,
        mean - margin,
        mean + margin
    )

def confidence_interval_corrected(
    values: np.ndarray,
    n_test: int,
    confidence: float = 0.95
    ):

    values = np.asarray(values)

    n = len(values)

    mean = np.mean(values)
    std = values.std(ddof=1)

    rho = 1.0 / n_test

    corrected_se = std * np.sqrt(
        (1.0 / n) + (rho / (1.0 - rho))
    )

    margin = corrected_se * t.ppf(
        (1 + confidence) / 2,
        n - 1
    )

    return (
        mean,
        mean - margin,
        mean + margin
    )

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
