import numpy as np

from src.preprocessing import (
    segment_tissue,
    normalize_staining
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
