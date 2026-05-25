import numpy as np
from src.features import extract_features
from src.preprocessing import segment_tissue


def test_extract_features_returns_array (sample_image):
    mask = segment_tissue(sample_image)

    features = extract_features(
        sample_image,
        mask,
        feature_set = "all"
    )

    assert isinstance(
        features,
        np.ndarray
    )


def test_extract_features_nonempty(sample_image):
    mask = segment_tissue(sample_image)

    features = extract_features(
        sample_image,
        mask,
        feature_set = "all"
    )

    assert len(features) > 0


def test_extract_features_no_nan(sample_image):
    mask = segment_tissue(sample_image)

    features = extract_features(
        sample_image,
        mask,
        feature_set = "all"
    )

    assert not np.isnan(features).any()
