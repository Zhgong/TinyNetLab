import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


def main() -> None:
    st.set_page_config(page_title="Moons Explorer")
    st.title("Moons 数据可视化")

    samples = st.slider("样本数", min_value=100, max_value=1000, value=800, step=50)
    noise = st.slider("噪声", min_value=0.0, max_value=0.5, value=0.2, step=0.01)

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
        title='Moons 数据集'
    )

    show_lr = st.checkbox("显示线性模型效果")

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
        st.write(f"Logistic Regression 损失: {loss:.4f}")
        st.markdown("线性模型只能学习直线决策边界，因此无法正确分开弯曲的 moons 数据。")
    else:
        st.altair_chart(chart, use_container_width=True)

    st.markdown(
        """
**思考**

- 画出来的两条“月牙”为什么一条直线分不开？
- 如果只有线性模型（如 logistic regression），损失大概会降到哪？也可以试试验证你的猜想。

提示：线性模型只能画出直线决策边界，而 moons 数据集呈现弯曲月牙形状，因此线性模型无法获得很低的损失。
"""
    )


if __name__ == "__main__":
    main()
