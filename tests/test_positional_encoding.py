import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from attention_demo import sinusoidal_positional_encoding


def test_sinusoidal_positional_encoding_shape_and_values():
    length, dim = 3, 4
    pe = sinusoidal_positional_encoding(length, dim)
    assert pe.shape == (length, dim)
    # Position 0 should be all zeros except cos terms which are 1
    assert np.allclose(pe[0, 0], 0.0)
    assert np.allclose(pe[0, 1], 1.0)
    if dim > 2:
        assert np.allclose(pe[0, 2], 0.0)
        assert np.allclose(pe[0, 3], 1.0)
