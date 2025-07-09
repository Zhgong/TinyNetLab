import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from tinynet import compute_loss


def test_compute_loss_basic():
    preds = np.array([[0.9], [0.2]])
    labels = np.array([[1], [0]])
    expected = -(
        labels * np.log(preds + 1e-8)
        + (1 - labels) * np.log(1 - preds + 1e-8)
    ).mean()
    assert compute_loss(preds, labels) == pytest.approx(expected)

