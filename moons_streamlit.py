import streamlit as st
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

st.set_page_config(page_title="Moons Explorer")
st.title("Moons \u6570\u636e\u53ef\u89c6\u5316")
st.set_option("deprecation.showPyplotGlobalUse", False)

samples = st.slider("\u6837\u672c\u6570", min_value=100, max_value=1000, value=800, step=50)
noise = st.slider("\u566a\u58f0", min_value=0.0, max_value=0.5, value=0.2, step=0.01)

X, y = make_moons(n_samples=samples, noise=noise, random_state=42)

fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=15)
ax.set_title("Moons \u6570\u636e\u5206\u5e03")
st.pyplot(fig)

st.markdown(
    """
**\u601d\u8003\u9898**

- \u201c\u753b\u51fa\u6765\u7684\u4e24\u6761\u2018\u6708\u7259\u2019\u4e3a\u4ec0\u4e48\u4e00\u6761\u76f4\u7ebf\u5206\u4e0d\u5f00\uff1f\u201d
- \u201c\u5982\u679c\u53ea\u6709\u7ebf\u6027\u6a21\u578b\uff08\u5982 logistic regression\uff09\uff0c\u635f\u5931\u5927\u6982\u4f1a\u964d\u5230\u54ea\uff1f\u4e5f\u53ef\u4ee5\u8bd5\u8bd5\u9a8c\u8bc1\u4f60\u7684\u731c\u60f3\u3002\u201d
"""
)

