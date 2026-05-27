import numpy as np
from src.statistics_utils import verify_same_splits
from sklearn.model_selection import RepeatedStratifiedKFold


def test_no_train_test_overlap():
    X = np.random.rand(100, 10)

    y = np.array(
        [0] * 50 +
        [1] * 50
    )

    rskf = RepeatedStratifiedKFold(
        n_splits = 5,
        n_repeats = 2,
        random_state = 42
    )

    for train_idx, test_idx in rskf.split(X, y):
        overlap = set(train_idx).intersection(
            set(test_idx)
        )

        assert len(overlap) == 0


def test_class_balance_preserved():
    X = np.random.rand(100, 10)

    y = np.array(
        [0] * 50 +
        [1] * 50
    )

    rskf = RepeatedStratifiedKFold(
        n_splits = 5,
        n_repeats = 2,
        random_state = 42
    )

    for train_idx, test_idx in rskf.split(X, y):
        y_train = y[train_idx]
        y_test = y[test_idx]

        assert abs(
            y_train.mean() -
            y_test.mean()
        ) < 0.1

def test_verify_same_splits():
    assert verify_same_splits(
        "cv_splits/100X_89_439_seed_42",
        "cv_splits/100X_89_439_seed_42"
    )

