import numpy as np
import tinynet as tn


def test_update_params():
    tn.init_params()

    old_W1 = tn.W1.copy()
    old_b1 = tn.b1.copy()
    old_W2 = tn.W2.copy()
    old_b2 = tn.b2.copy()

    grads = [
        np.full_like(tn.W1, 0.5),
        np.full_like(tn.b1, -0.2),
        np.full_like(tn.W2, 1.5),
        np.full_like(tn.b2, -0.3),
    ]
    lr = 0.05

    tn.update_params(grads, lr=lr)

    assert np.allclose(tn.W1, old_W1 - lr * grads[0])
    assert np.allclose(tn.b1, old_b1 - lr * grads[1])
    assert np.allclose(tn.W2, old_W2 - lr * grads[2])
    assert np.allclose(tn.b2, old_b2 - lr * grads[3])
