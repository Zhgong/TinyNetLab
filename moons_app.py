import streamlit as st
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from i18n import t


def main() -> None:
    st.set_page_config(page_title=t("moons_explore_page_title"), page_icon="🌙")

    st.title(t("moons_distribution_title"))

    st.markdown(t("moons_think"))

    samples = st.slider(t("samples"), min_value=100, max_value=1000, value=800, step=50)
    noise = st.slider(t("noise"), min_value=0.0, max_value=0.5, value=0.2, step=0.01)

    X, y = make_moons(n_samples=samples, noise=noise, random_state=42)

    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=15)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(t("moons_distribution_title"))

    st.pyplot(fig)


if __name__ == "__main__":
    main()
