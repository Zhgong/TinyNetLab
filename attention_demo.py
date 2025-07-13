"""Interactive attention mechanism demo using Streamlit and Altair."""

from __future__ import annotations

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from typing import Union
from math import sqrt

from i18n import t

# Allow large heatmaps
alt.data_transformers.disable_max_rows()

# Predefined short sentences
SENTENCES = [
    "I love machine learning",
    "Attention is all you need",
    "Tiny models can be fun",
]


def sinusoidal_positional_encoding(length: int, dim: int) -> np.ndarray:
    """Return standard sinusoidal positional encoding."""
    position = np.arange(length)[:, None]
    div_term = np.exp(np.arange(0, dim, 2) * -(np.log(10000.0) / dim))
    pe = np.zeros((length, dim))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe


def positional_encoding_chart(pe: np.ndarray) -> alt.LayerChart:
    """Visualize positional encoding values as a heatmap."""
    df = pd.DataFrame(pe)
    df["pos"] = np.arange(pe.shape[0])
    df = df.melt(id_vars="pos", var_name="dim", value_name="value")
    base = alt.Chart(df).encode(x="dim", y=alt.Y("pos:O", sort=None))
    heat = base.mark_rect().encode(color=alt.Color("value", scale=alt.Scale(scheme="viridis")))
    text = base.mark_text().encode(text=alt.Text("value", format=".2f"))
    return (heat + text).properties(width=400, height=200, title=t("positional_encoding"))

def init_state(dim: int, tokens: list[str]) -> None:
    """Initialize vectors in session state."""
    if "vectors" not in st.session_state or st.session_state.get("dim") != dim or st.session_state.get("tokens") != tokens:
        rng = np.random.default_rng(0)
        st.session_state["vectors"] = rng.normal(0, 1, (len(tokens), dim))
        st.session_state["dim"] = dim
        st.session_state["tokens"] = tokens


def get_vectors(dim: int, tokens: list[str]) -> np.ndarray:
    init_state(dim, tokens)
    vecs = st.session_state["vectors"]
    for i, tok in enumerate(tokens):
        cols = st.columns(dim)
        for j in range(dim):
            key = f"vec_{i}_{j}"
            vecs[i, j] = cols[j].number_input(
                f"{tok}[{j}]", value=float(vecs[i, j]), key=key
            )
    st.session_state["vectors"] = vecs
    return vecs


def attention(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute scaled dot-product attention."""
    scores = q @ k.T / sqrt(q.shape[1])
    weights = np.exp(scores - scores.max(axis=1, keepdims=True))
    weights = weights / weights.sum(axis=1, keepdims=True)
    output = weights @ v
    return scores, weights, output


def heatmap_chart(tokens: list[str], scores: np.ndarray, weights: np.ndarray) -> alt.LayerChart:
    """Return an interactive heatmap for attention weights."""
    n = len(tokens)
    df = pd.DataFrame(
        {
            "query": np.repeat(tokens, n),
            "key": np.tile(tokens, n),
            "score": scores.flatten(),
            "weight": weights.flatten(),
        }
    )
    base = alt.Chart(df).encode(x="key", y="query")
    heat = base.mark_rect().encode(color=alt.Color("weight", scale=alt.Scale(scheme="blues")))
    text = base.mark_text().encode(text=alt.Text("weight", format=".2f"))
    tooltip = base.mark_rect(opacity=0).encode(
        tooltip=["query", "key", alt.Tooltip("score", format=".2f"), alt.Tooltip("weight", format=".2f")]
    )
    return (heat + text + tooltip).properties(width=300, height=300, title=t("attention_matrix"))


def vector_chart(tokens: list[str], x: np.ndarray, out: np.ndarray) -> Union[alt.LayerChart,alt.Chart]:
    """Show 2D vectors before and after attention."""
    if x.shape[1] < 2:
        return alt.Chart(pd.DataFrame()).mark_point()
    df = pd.DataFrame({
        "x1": x[:, 0],
        "x2": x[:, 1],
        "o1": out[:, 0],
        "o2": out[:, 1],
        "token": tokens,
    })
    points = alt.Chart(df).mark_point(color="blue", size=60).encode(x="x1", y="x2", tooltip=["token"])
    out_points = alt.Chart(df).mark_point(color="red", size=60).encode(x="o1", y="o2")
    lines = alt.Chart(df).mark_line(color="gray").encode(x="x1", y="x2", x2="o1", y2="o2")
    return (points + out_points + lines).properties(width=400, height=400, title=t("vector_transform"))


def main() -> None:
    st.set_page_config(page_title=t("attention_demo"))
    st.title(t("attention_demo"))

    sentence = st.selectbox(t("select_sentence"), SENTENCES)
    tokens = sentence.split()

    dim = st.slider(t("dimension"), 2, 4, 2, step=1)

    if st.button(t("randomize_vectors")):
        rng = np.random.default_rng()
        st.session_state["vectors"] = rng.normal(0, 1, (len(tokens), dim))
        st.session_state["dim"] = dim
        st.session_state["tokens"] = tokens

    X = get_vectors(dim, tokens)

    use_pe = st.checkbox(t("use_positional_encoding"), value=False)
    show_pe = st.checkbox(t("show_positional_encoding"), value=False)
    with st.expander(t("positional_encoding_help_title")):
        st.markdown(t("positional_encoding_help"))
    if use_pe or show_pe:
        pe = sinusoidal_positional_encoding(len(tokens), dim)
        if show_pe:
            st.altair_chart(positional_encoding_chart(pe), use_container_width=True)
        if use_pe:
            X = X + pe

    W_Q = np.eye(dim)
    W_K = np.eye(dim)
    W_V = np.eye(dim)
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V

    scores, weights, output = attention(Q, K, V)

    if st.checkbox(t("show_attention_matrix"), value=True):
        st.altair_chart(heatmap_chart(tokens, scores, weights), use_container_width=True)
        with st.expander(t("attention_matrix_help_title")):
            st.markdown(t("attention_matrix_help"))

    st.altair_chart(vector_chart(tokens, X, output), use_container_width=True)

    with st.expander(t("vector_chart_help_title")):
        st.markdown(t("vector_chart_help"))


if __name__ == "__main__":
    main()
