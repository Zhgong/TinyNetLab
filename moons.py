from sklearn.datasets import make_moons
import numpy as np
import matplotlib.pyplot as plt


def main() -> None:
    X, y = make_moons(n_samples=800, noise=0.2, random_state=42)
    y = y.reshape(-1, 1)  # (800,1) 方便矩阵运算

    plt.scatter(X[:, 0], X[:, 1], c=y[:, 0], cmap="coolwarm", s=15)
    plt.title("Moons 数据分布")
    plt.show()


if __name__ == "__main__":
    main()
