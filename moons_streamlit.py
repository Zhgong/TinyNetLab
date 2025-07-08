import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from i18n import t


def main() -> None:
    st.set_page_config(page_title=t("moons_explore_page_title"))
    st.title(t("moons_title"))

    samples = st.slider(t("samples"), min_value=100, max_value=1000, value=800, step=50)
    noise = st.slider(t("noise"), min_value=0.0, max_value=0.5, value=0.2, step=0.01)

    X, y = make_moons(n_samples=samples, noise=noise, random_state=42)

    # 构造 DataFrame 以便 Altair 使用
    df = pd.DataFrame({
        'x1': X[:, 0],
        'x2': X[:, 1],
        'label': y
    })

    # 用 Altair 画散点图
    chart = alt.Chart(df).mark_circle(size=60, opacity=1).encode(
        x='x1',
        y='x2',
        color=alt.Color('label:N', scale=alt.Scale(scheme='dark2')),
        tooltip=['x1', 'x2', 'label']
    ).properties(
        width=600,
        height=400,
        title=t('moons_dataset')
    )

    show_lr = st.checkbox(t("show_linear_model"))

    if show_lr:
        model = LogisticRegression(max_iter=200)
        model.fit(X, y)

        proba = model.predict_proba(X)[:, 1]
        loss = log_loss(y, proba)

        w1, w2 = model.coef_[0]
        b = model.intercept_[0]
        x_vals = np.array([df.x1.min(), df.x1.max()])
        y_vals = -(w1 * x_vals + b) / w2
        boundary_df = pd.DataFrame({'x1': x_vals, 'x2': y_vals})
        line = alt.Chart(boundary_df).mark_line(color='black').encode(x='x1', y='x2')

        st.altair_chart(chart + line, use_container_width=True)
        st.write(t("log_reg_loss", loss=loss))
        st.markdown(t("linear_model_note"))
    else:
        st.altair_chart(chart, use_container_width=True)

    st.markdown(t("moons_think"))


if __name__ == "__main__":
    main()
