"""Tiny 2-4-1 neural network with Streamlit visualization."""

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from sklearn.datasets import make_moons

# Initialize random generator and network dimensions
rng = np.random.default_rng(42)
n_input, n_hidden, n_output = 2, 4, 1

# Parameters will be initialized in ``init_params`` so that each training run
# starts fresh.
W1: np.ndarray
b1: np.ndarray
W2: np.ndarray
b2: np.ndarray


def init_params() -> None:
    """Initialize network parameters."""
    global W1, b1, W2, b2
    W1 = rng.normal(0, 0.1, (n_input, n_hidden))
    b1 = np.zeros((1, n_hidden))
    W2 = rng.normal(0, 0.1, (n_hidden, n_output))
    b2 = np.zeros((1, n_output))

# Activation functions

def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)

def d_relu(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(float)

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))

# Forward pass

def forward(X: np.ndarray):
    Z1 = X @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    A2 = sigmoid(Z2)
    cache = (X, Z1, A1, Z2, A2)
    return A2, cache

# Loss (binary cross entropy)

def compute_loss(A2: np.ndarray, y: np.ndarray) -> float:
    eps = 1e-8
    return -(y * np.log(A2 + eps) + (1 - y) * np.log(1 - A2 + eps)).mean()

# Backward propagation

def backward(cache, y: np.ndarray):
    X, Z1, A1, Z2, A2 = cache
    m = X.shape[0]

    dZ2 = A2 - y
    dW2 = (A1.T @ dZ2) / m
    db2 = dZ2.mean(axis=0, keepdims=True)

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * d_relu(Z1)
    dW1 = (X.T @ dZ1) / m
    db1 = dZ1.mean(axis=0, keepdims=True)

    return dW1, db1, dW2, db2

# Gradient descent update

def update_params(grads, lr: float = 0.1):
    global W1, b1, W2, b2
    dW1, db1, dW2, db2 = grads
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

# Training loop

def train(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 2000,
    lr: float = 0.1,
    progress: "st.progress" = None,
) -> float:
    """Train the network and optionally update a Streamlit progress bar."""

    for i in range(epochs):
        A2, cache = forward(X)
        loss = compute_loss(A2, y)
        grads = backward(cache, y)
        update_params(grads, lr)

        if progress and i % max(1, epochs // 100) == 0:
            progress.progress((i + 1) / epochs)

    return float(loss)

# Visualization of decision boundary

def decision_boundary_chart(X: np.ndarray, y: np.ndarray) -> alt.Chart:
    """Return an Altair chart visualizing the decision boundary."""

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs, _ = forward(grid)

    df_grid = pd.DataFrame({"x1": grid[:, 0], "x2": grid[:, 1], "prob": probs[:, 0]})
    df_data = pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "label": y[:, 0]})

    boundary = (
        alt.Chart(df_grid)
        .mark_rect()
        .encode(
            x=alt.X("x1:Q", bin=False),
            y=alt.Y("x2:Q", bin=False),
            color=alt.Color(
                "prob:Q",
                scale=alt.Scale(scheme="blueorange", domain=[0.0, 1.0]),
                legend=None,
            ),
        )
    )
    points = (
        alt.Chart(df_data)
        .mark_circle(size=35, opacity=1)
        .encode(
            x="x1",
            y="x2",
            color=alt.Color("label:N", scale=alt.Scale(scheme="dark2")),
        )
    )

    return (boundary + points).properties(width=600, height=400, title="Decision Boundary")


def main() -> None:
    """Streamlit UI for training and visualizing the tiny network."""

    st.set_page_config(page_title="TinyNet Trainer")
    st.title("TinyNet 2-4-1")
    st.markdown(
        "Train a tiny neural network on the moons dataset and view the decision boundary."
    )

    samples = st.slider("Samples", 100, 1000, 800, step=50)
    noise = st.slider("Noise", 0.0, 0.5, 0.2, step=0.01)
    epochs = st.slider("Epochs", 500, 5000, 2000, step=100)
    lr = st.slider("Learning rate", 0.01, 1.0, 0.1, step=0.01)

    if st.button("Train"):
        X, y = make_moons(n_samples=samples, noise=noise, random_state=42)
        y = y.reshape(-1, 1)

        init_params()
        progress = st.progress(0.0)
        loss = train(X, y, epochs=epochs, lr=lr, progress=progress)

        st.write(f"Final loss: {loss:.4f}")
        st.altair_chart(decision_boundary_chart(X, y), use_container_width=True)


if __name__ == "__main__":
    main()
