import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from tinynet import relu, d_relu, sigmoid


def test_relu_behavior():
    data = np.array([-2.0, -0.1, 0.0, 0.5, 3.0])
    result = relu(data)
    expected = np.array([0.0, 0.0, 0.0, 0.5, 3.0])
    assert np.allclose(result, expected)


def test_d_relu_behavior():
    data = np.array([-2.0, -0.1, 0.0, 0.5, 3.0])
    result = d_relu(data)
    expected = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
    assert np.allclose(result, expected)


def test_sigmoid_known_values():
    assert np.allclose(sigmoid(0), 0.5)
    assert np.allclose(sigmoid(1), 1 / (1 + np.exp(-1)))
    assert np.allclose(sigmoid(-1), 1 / (1 + np.exp(1)))

