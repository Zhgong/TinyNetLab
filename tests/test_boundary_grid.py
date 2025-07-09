import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from tinynet import create_boundary_grid

def test_create_boundary_grid_shape_and_columns():
    X = np.array([[0, 0], [1, 1], [2, 2]])
    grid, df_grid = create_boundary_grid(X)
    assert grid.shape == (300 * 300, 2)
    assert list(df_grid.columns) == ["x1", "x2"]
    assert len(df_grid) == grid.shape[0]
