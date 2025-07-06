import streamlit as st
import pandas as pd
import altair as alt
from sklearn.datasets import make_moons

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

st.altair_chart(chart, use_container_width=True)

st.markdown(
"""
**思考**

- 画出来的两条“月牙”为什么一条直线分不开？
- 如果只有线性模型（如 logistic regression），损失大概会降到哪？也可以试试验证你的猜想。
"""
)
