import os
import numpy as np

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

        same = np.array_equal(exp1_test, exp2_test)

        print(f"Fold {fold}: {'MATCH' if same else 'DIFFERENT'}")
