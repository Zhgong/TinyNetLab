import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tinynet as tn


def test_forward_backward_shapes_and_grads():
    # Initialize deterministic parameters
    tn.init_params()

    # Small deterministic dataset
    X = np.array([[0.1, -0.2], [0.3, 0.0]], dtype=float)
    y = np.array([[1.0], [0.0]])

    # Forward pass
    A2, cache = tn.forward(X)

    # Verify output shape
    assert A2.shape == (X.shape[0], tn.n_output)

    # Backward pass
    dW1, db1, dW2, db2 = tn.backward(cache, y)

    # Gradient shapes should match parameter shapes
    assert dW1.shape == tn.W1.shape
    assert db1.shape == tn.b1.shape
    assert dW2.shape == tn.W2.shape
    assert db2.shape == tn.b2.shape

    # Numerical gradient check for correctness
    eps = 1e-7

    def num_grad(param):
        grad = np.zeros_like(param)
        it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            orig = param[idx]
            param[idx] = orig + eps
            plus_loss = tn.compute_loss(tn.forward(X)[0], y)
            param[idx] = orig - eps
            minus_loss = tn.compute_loss(tn.forward(X)[0], y)
            param[idx] = orig
            grad[idx] = (plus_loss - minus_loss) / (2 * eps)
            it.iternext()
        return grad

    num_dW1 = num_grad(tn.W1)
    num_db1 = num_grad(tn.b1)
    num_dW2 = num_grad(tn.W2)
    num_db2 = num_grad(tn.b2)

    assert np.allclose(dW1, num_dW1, atol=1e-6, rtol=1e-5)
    assert np.allclose(db1, num_db1, atol=1e-6, rtol=1e-5)
    assert np.allclose(dW2, num_dW2, atol=1e-6, rtol=1e-5)
    assert np.allclose(db2, num_db2, atol=1e-6, rtol=1e-5)
