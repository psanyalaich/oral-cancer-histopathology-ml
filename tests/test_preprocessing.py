import numpy as np

from src.preprocessing import (
    segment_tissue,
    normalize_staining,
    create_reinhard_normalizer
)

def test_segment_tissue_returns_mask(sample_image):
    mask = segment_tissue(sample_image)
    assert mask.shape == (512, 512)
    assert mask.dtype == np.uint8


def test_segment_tissue_binary_mask(sample_image):
    mask = segment_tissue(sample_image)
    unique_values = np.unique(mask)
    assert set(unique_values).issubset({0, 255})


def test_normalize_staining_none(sample_image):
    normalized = normalize_staining(
        sample_image,
        method = None
    )

    assert normalized.shape == (512, 512, 3)

def test_reinhard_changes_image(sample_image):
    normalizer = create_reinhard_normalizer(sample_image)

    normalized = normalize_staining(
        sample_image,
        method = "reinhard",
        normalizer = normalizer
    )

    assert normalized.shape == sample_image.shape

    assert not np.array_equal(
        normalized,
        sample_image
    )
