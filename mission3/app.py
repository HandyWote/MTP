"""Streamlit 交互仪表盘（加分项）。复用 bootstrap_core，不改报告图。
运行：streamlit run mission3/app.py（从仓库根）或 streamlit run app.py（从 mission3/）"""
import sys
from pathlib import Path
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mission3 import viz_style; viz_style.apply()
from mission3.bootstrap_core import load_aircondit7, bootstrap_medians, percentile_ci, bca_ci

st.set_page_config(page_title="Bootstrap 中位数", layout="wide")
st.sidebar.header("参数")
data_src = st.sidebar.radio("数据源", ["aircondit7（真实 n=24）", "Exp(1) 模拟"])
n = st.sidebar.slider("样本量 n（仅 Exp(1) 生效）", 12, 200, 24, 2)
B = st.sidebar.slider("重采样次数 B", 100, 20000, 10000, 100)
seed = st.sidebar.number_input("随机种子", 0, 9999, 42)

# “重新重采样”按钮：每次点击改变有效种子，从而抽出新的 bootstrap 样本
if "clicks" not in st.session_state:
    st.session_state.clicks = 0
if st.sidebar.button("重新重采样"):
    st.session_state.clicks += 1

rng = np.random.default_rng(int(seed) + st.session_state.clicks)
if data_src.startswith("aircondit7"):
    x = load_aircondit7()           # 真实数据固定 n=24，无法扩样
else:
    x = rng.exponential(size=n)     # Exp(1)，n 由滑块控制；真中位数 ln2≈0.693

m_hat, boot = bootstrap_medians(x, B, rng)
plo, phi = percentile_ci(boot)
blo, bhi, z0, a_hat = bca_ci(x, boot)

st.title("Bootstrap 估计中位数 · 交互演示")
st.caption(f"当前样本量 = {len(x)}，B = {B}")

tab1, tab2, tab3 = st.tabs(["原始 + Bootstrap 分布", "CI 对比", "说明"])
with tab1:
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots()
        ax.hist(x, bins=15, alpha=0.6, color="C0")
        ax.axvline(np.median(x), color="C1", ls="--", label=f"中位数 {np.median(x):.2f}")
        ax.set_title("原始样本"); ax.set_xlabel("值"); ax.legend()
        st.pyplot(fig)
    with c2:
        fig2, ax2 = plt.subplots()
        ax2.hist(boot, bins=40, alpha=0.6, color="C2")
        ax2.axvline(plo, color="r", label="percentile CI")
        ax2.axvline(phi, color="r")
        ax2.axvline(blo, color="b", ls="--", label="BCa CI")
        ax2.axvline(bhi, color="b", ls="--")
        ax2.set_title("Bootstrap 中位数分布"); ax2.set_xlabel("中位数"); ax2.legend()
        st.pyplot(fig2)
with tab2:
    st.write(f"点估计 $\\hat m$ = **{m_hat:.3f}**")
    st.write(f"percentile 95% CI = [{plo:.2f}, {phi:.2f}]")
    st.write(f"BCa 95% CI = [{blo:.2f}, {bhi:.2f}]  (z0={z0:.3f}, a_hat={a_hat:.4f})")
    st.caption("注：中位数偶数 n 下 a_hat≈0，BCa≈percentile（设计 §4.4）。")
with tab3:
    st.markdown(
        "方法/假设/为何用 Bootstrap 见《任务三设计.md》。\n\n"
        "本仪表盘为**加分项**，复用 `bootstrap_core.py`；报告图仍用 matplotlib（`plot_matplotlib.py`）。"
    )
