from src.statistics_utils import verify_same_splits

same = verify_same_splits(
    "cv_splits/100X_89_439",
    "cv_splits/100X_89_439",
    )

print("\nOverall:", "MATCH" if same else "DIFFERENT")
