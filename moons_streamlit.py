import streamlit as st
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

st.set_page_config(page_title="Moons Explorer")
st.title("Moons 数据可视化")
# st.set_option("deprecation.showPyplotGlobalUse", False)

samples = st.slider("样本数", min_value=100, max_value=1000, value=800, step=50)
noise = st.slider("噪声", min_value=0.0, max_value=0.5, value=0.2, step=0.01)

X, y = make_moons(n_samples=samples, noise=noise, random_state=42)

fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=15)
ax.set_title("Moons 数据集")
st.pyplot(fig)

st.markdown(
"""
**思考**

- 画出来的两条“月牙”为什么一条直线分不开？
- 如果只有线性模型（如 logistic regression），损失大概会降到哪？也可以试试验证你的猜想。

"""
)

