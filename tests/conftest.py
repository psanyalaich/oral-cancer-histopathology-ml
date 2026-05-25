import sys
import pytest
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(PROJECT_ROOT))

@pytest.fixture
def sample_image():
    rng = np.random.default_rng(42)

    img = rng.integers(
        0,
        255,
        size=(512, 512, 3),
        dtype=np.uint8
    )

    return img
