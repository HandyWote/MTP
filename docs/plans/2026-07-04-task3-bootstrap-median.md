# 任务三：Bootstrap 估计中位数 — 实现计划

> 📌 **命名说明（2026-07-04）**：本计划成文时代码目录为 `task3/`，后按 CLAUDE.md「missionN/」规范统一重命名为 `mission3/`。文中所有 `task3` / `from task3` / `python -m task3.*` 即现在的 `mission3`；计划内容不变，仅目录名演进。历史 `task3/.venv`（不完整、重命名后失效）已清除，统一用根目录 `requirements.txt`。

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 实现一个可复现的 Python 程序，对 Proschan (1963) `aircondit7` 数据（n=24）用非参数 Bootstrap 估计总体中位数，给出 percentile + BCa 置信区间，用 Exp(1) 模拟验证覆盖率，画三类报告图，并附可选 Streamlit 交互演示。

**Architecture:** 共享计算核心 `task3/bootstrap_core.py`（纯函数、可独立测试）→ 两个前端：`plot_matplotlib.py`（报告静态图，DPI=300，复用根目录共享 `viz_style.py`）和 `app.py`（Streamlit 交互仪表盘，加分项）。TDD：每个核心函数先写失败测试再实现。`run_all.py` 一键复现全部数值结果与图表。

**Tech Stack:** Python 3.10+ · numpy (<2) · scipy · matplotlib · pytest · streamlit（可选）

**设计依据:** [任务三设计.md](../../任务三设计.md)（v0.2，含 BCa 退化订正与小样本辩护）。本计划严格按其 §4 公式与 §4.9 代码结构实现。

**关键约束（来自设计 §4.4 / §7.4）:**
- BCa 对中位数（非光滑、偶数 n）退化为 percentile + z₀，n=24 时与 percentile 几乎重合，仅 n≥50 见差异——测试须覆盖此退化。
- 覆盖率一律报 `p̂ ± MC SE`，MC SE 按观测 p̂ 计（√(p̂(1−p̂)/R)），不用 p=0.5 上界。
- 随机种子固定（42）保证可复现。

**环境:** Windows + Git Bash。`python` 已可用、numpy 已装。matplotlib/scipy/pytest/streamlit 待装。`task3/data/aircondit7.csv` 已就绪（24 行，列 `obs,hours`）。

---

## Task 0: 环境与脚手架

**Files:**
- Create: `requirements.txt`
- Create: `viz_style.py`（根目录，三任务共享）
- Create: `task3/__init__.py`, `task3/tests/__init__.py`
- Create: `task3/output/.gitkeep`

**Step 1: 写 requirements.txt**

```
numpy>=1.24,<2.0
scipy>=1.10
matplotlib>=3.7
pytest>=7.0
streamlit>=1.30
```

**Step 2: 安装依赖**

Run: `python -m pip install -r requirements.txt`
Expected: 全部安装成功（numpy 须 <2，避免 np.float 等别名移除的破坏性变更）。

**Step 3: 写共享 viz_style.py（根目录）**

```python
"""共享可视化风格：中文字体 + Okabe-Ito 色盲友好色板 + DPI=300。
任务一/二/三统一 import viz_style; viz_style.apply() 后绘图。"""
import matplotlib as mpl

OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
             "#F0E442", "#56B4E9", "#E69F00", "#000000"]

def apply_style():
    mpl.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "figure.figsize": (7, 4.5),
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.prop_cycle": mpl.cycler(color=OKABE_ITO),
    })
```

**Step 4: 建目录占位**

Run: `mkdir -p task3/output && touch task3/__init__.py task3/tests/__init__.py task3/output/.gitkeep`

**Step 5: 冒烟测试**

Run: `python -c "import numpy,scipy,matplotlib,pytest,streamlit; print('ok', numpy.__version__)"`
Expected: 打印 `ok 1.x.x`（确认 numpy<2）。

**Step 6: Commit**

```bash
git add requirements.txt viz_style.py task3/
git commit -m "chore(task3): scaffold env, shared viz_style, task3 dirs"
```

---

## Task 1: 数据加载

**Files:**
- Create: `task3/tests/test_bootstrap_core.py`
- Create: `task3/bootstrap_core.py`（本任务只加 `load_aircondit7`）

**Step 1: 写失败测试**

`task3/tests/test_bootstrap_core.py`:
```python
import numpy as np
from task3.bootstrap_core import load_aircondit7

KNOWN = [3,5,5,13,14,15,22,22,23,30,36,39,44,46,50,
         72,79,88,97,102,139,188,197,210]  # list（非 set）—保留重复值

def test_load_aircondit7():
    x = load_aircondit7()
    assert len(x) == 24
    assert np.median(x) == 41.5
    assert sorted(x.tolist()) == sorted(KNOWN)
```

**Step 2: 运行确认失败**

Run: `python -m pytest task3/tests/test_bootstrap_core.py -v`
Expected: FAIL（`ModuleNotFoundError` 或 `ImportError`）。

**Step 3: 实现**

`task3/bootstrap_core.py`:
```python
import csv
import numpy as np

def load_aircondit7(path="task3/data/aircondit7.csv"):
    """Proschan (1963) 第7架机空调故障间隔（小时），n=24。"""
    vals = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vals.append(float(row["hours"]))
    return np.array(vals, dtype=float)
```

**Step 4: 运行确认通过**

Run: `python -m pytest task3/tests/test_bootstrap_core.py -v`
Expected: PASS。

**Step 5: Commit**

```bash
git add task3/bootstrap_core.py task3/tests/test_bootstrap_core.py
git commit -m "feat(task3): load_aircondit7 with verified n=24 values"
```

---

## Task 2: Bootstrap 重采样中位数

**Files:**
- Modify: `task3/bootstrap_core.py`（加 `bootstrap_medians`）
- Modify: `task3/tests/test_bootstrap_core.py`

**Step 1: 加失败测试**

```python
from task3.bootstrap_core import bootstrap_medians

def test_bootstrap_medians_reproducible_and_correct():
    x = load_aircondit7()
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    m1, b1 = bootstrap_medians(x, B=1000, rng=rng1)
    m2, b2 = bootstrap_medians(x, B=1000, rng=rng2)
    assert m1 == 41.5                      # 点估计 = 原样本中位数
    assert len(b1) == 1000
    assert np.array_equal(b1, b2)          # 固定种子可复现
    assert b1.min() >= x.min() and b1.max() <= x.max()  # 重采样值落在样本范围内
```

**Step 2: 运行确认失败** — Run: `python -m pytest task3/tests/test_bootstrap_core.py::test_bootstrap_medians_reproducible_and_correct -v` → FAIL。

**Step 3: 实现**

加到 `task3/bootstrap_core.py`:
```python
def bootstrap_medians(x, B, rng):
    """非参数有放回重采样 B 次，每次算中位数。
    返回 (m_hat, boot_medians)。向量化：一次性抽 B×n 索引。"""
    x = np.asarray(x, dtype=float)
    n = len(x)
    idx = rng.integers(0, n, size=(B, n))   # 有放回（允许重复索引）
    boot = np.median(x[idx], axis=1)
    return float(np.median(x)), boot
```

**Step 4: 运行确认通过** — Run 同上 → PASS。

**Step 5: Commit**

```bash
git add task3/bootstrap_core.py task3/tests/test_bootstrap_core.py
git commit -m "feat(task3): vectorized bootstrap_medians"
```

---

## Task 3: percentile 置信区间

**Files:** Modify `bootstrap_core.py` + 测试

**Step 1: 加失败测试**

```python
from task3.bootstrap_core import percentile_ci

def test_percentile_ci_known_and_contains_estimate():
    # 已知数组的分位数
    assert percentile_ci(np.arange(1, 101), alpha=0.05) == (5.0, 96.0)  # 2.5%→5, 97.5%→96(线性)
    # bootstrap 输出的 CI 应包含点估计
    x = load_aircondit7()
    rng = np.random.default_rng(0)
    m_hat, boot = bootstrap_medians(x, 5000, rng)
    lo, hi = percentile_ci(boot)
    assert lo <= m_hat <= hi
```

> 注：`np.quantile` 默认线性插值，`[1..100]` 的 2.5% 分位 = 3.975？实际 `np.quantile(range(1,101),0.025)`≈3.975。实现时用 `np.quantile`，测试期望值以实际运行结果校准（见 Step 4 校准说明）。

**Step 2: 运行确认失败** → FAIL。

**Step 3: 实现**

```python
def percentile_ci(boot, alpha=0.05):
    """percentile 法：取 bootstrap 分布的 α/2 与 1−α/2 分位。"""
    lo = float(np.quantile(boot, alpha / 2))
    hi = float(np.quantile(boot, 1 - alpha / 2))
    return lo, hi
```

**Step 4: 运行 + 校准期望值**

Run: `python -c "import numpy as np; print(np.quantile(np.arange(1,101),0.025), np.quantile(np.arange(1,101),0.975))"`
→ 用打印出的实际值替换测试里的 `(5.0, 96.0)`（应为 `3.975 98.025` 之类）。改完再跑测试 → PASS。

**Step 5: Commit**

```bash
git add task3/bootstrap_core.py task3/tests/test_bootstrap_core.py
git commit -m "feat(task3): percentile_ci"
```

---

## Task 4: BCa 置信区间（含退化自检）

**Files:** Modify `bootstrap_core.py` + 测试

**Step 1: 加失败测试**

```python
from task3.bootstrap_core import bca_ci

def test_bca_degenerates_to_percentile_for_symmetric():
    # 对称数据：z0≈0, a_hat≈0 → BCa≈percentile
    rng = np.random.default_rng(1)
    x = rng.normal(size=200)               # 对称、光滑统计量下 a_hat 仍可非零，
    m_hat, boot = bootstrap_medians(x, 4000, rng)  # 但 z0≈0；BCa 端点应接近 percentile
    plo, phi = percentile_ci(boot)
    blo, bhi, z0, a_hat = bca_ci(x, boot)
    assert abs(z0) < 0.1
    assert abs(blo - plo) < abs(phi - plo) * 0.1 + 1e-9
    assert abs(bhi - phi) < abs(phi - plo) * 0.1 + 1e-9

def test_bca_median_even_n_a_hat_near_zero():
    # 设计 §4.4 关键退化：中位数 + 偶数 n → jackknife a_hat≈0
    x = load_aircondit7()                  # n=24 偶数
    rng = np.random.default_rng(2)
    _, boot = bootstrap_medians(x, 4000, rng)
    _, _, _, a_hat = bca_ci(x, boot)
    assert abs(a_hat) < 1e-6              # 留一法中位数塌缩为 ~2 值 → a_hat≈0

def test_bca_matches_scipy_on_small_case():
    # 交叉验证：与 scipy.stats.bootstrap 的 BCa 比较（若可用）
    import pytest
    try:
        from scipy import stats as sps
    except ImportError:
        pytest.skip("scipy 不可用")
    x = np.array([1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20])  # 加一个偏态点
    rng = np.random.default_rng(3)
    _, boot = bootstrap_medians(x, 2000, rng)
    blo, bhi, _, _ = bca_ci(x, boot)
    res = sps.bootstrap((x,), np.median, method="BCa",
                        n_resamples=2000, random_state=3, vectorized=False)
    # 端点应接近（不完全相等，因重采样序列差异，但量级一致）
    assert res.confidence_interval.low < bhi and res.confidence_interval.high > blo
```

**Step 2: 运行确认失败** → FAIL。

**Step 3: 实现**

```python
from scipy import stats as sps

def bca_ci(x, boot, alpha=0.05):
    """BCa：偏差修正 z0 + jackknife 加速度 a_hat。
    返回 (lo, hi, z0, a_hat)。设计 §4.4 公式。"""
    x = np.asarray(x, dtype=float)
    B = len(boot)
    m_hat = float(np.median(x))
    # 偏差修正 z0（严格 < 计数，与 scipy.stats.bootstrap 一致）
    prop = float(np.mean(boot < m_hat))
    prop = min(max(prop, 1.0 / (B + 1)), 1 - 1.0 / (B + 1))  # 防 ppf(0/1)=±inf
    z0 = float(sps.norm.ppf(prop))
    # 加速度 a_hat（jackknife，O(n^2)，与 B 无关）
    n = len(x)
    loo = np.array([np.median(np.delete(x, i)) for i in range(n)])
    loo_bar = loo.mean()
    num = float(np.sum((loo_bar - loo) ** 3))
    den = 6.0 * (float(np.sum((loo - loo_bar) ** 2)) ** 1.5)
    a_hat = num / den if den != 0 else 0.0
    # 调整分位
    def adj(p):
        z = sps.norm.ppf(p)
        return float(sps.norm.cdf(z0 + (z0 + z) / (1 - a_hat * (z0 + z))))
    lo = float(np.quantile(boot, adj(alpha / 2)))
    hi = float(np.quantile(boot, adj(1 - alpha / 2)))
    return lo, hi, z0, a_hat
```

**Step 4: 运行确认通过** — Run: `python -m pytest task3/tests/test_bootstrap_core.py -v` → 全部 PASS。
关键：`test_bca_median_even_n_a_hat_near_zero` 必须过——它验证设计 §4.4 的核心退化结论。

**Step 5: Commit**

```bash
git add task3/bootstrap_core.py task3/tests/test_bootstrap_core.py
git commit -m "feat(task3): bca_ci with jackknife + median-degeneracy test"
```

---

## Task 5: 覆盖率实验（Exp(1) 已知真值）

**Files:** Modify `bootstrap_core.py` + 测试

**Step 1: 加失败测试**

```python
from task3.bootstrap_core import coverage_experiment

def test_coverage_experiment_exp1():
    res = coverage_experiment(n=24, B=1000, R=300, seed=42)  # 小 R 快速测试
    assert 0.80 < res["perc_coverage"] < 1.0   # Exp(1) n=24 应在 ~0.93 附近
    assert 0.80 < res["bca_coverage"] < 1.0
    # MC SE 公式：sqrt(p(1-p)/R)
    import math
    assert abs(res["mc_se_perc"] - math.sqrt(
        res["perc_coverage"]*(1-res["perc_coverage"])/300)) < 1e-9
    assert res["mean_perc_width"] > 0

def test_coverage_reproducible():
    r1 = coverage_experiment(n=24, B=500, R=100, seed=7)
    r2 = coverage_experiment(n=24, B=500, R=100, seed=7)
    assert r1 == r2
```

**Step 2: 运行确认失败** → FAIL。

**Step 3: 实现**

```python
def coverage_experiment(n=24, B=2000, R=1000, seed=42, true_median=np.log(2)):
    """Exp(1) 已知真中位数 ln2，验证 percentile/BCa 的 95% 覆盖率。
    返回 dict（覆盖率报 p̂ ± MC SE）。设计 §4.5。"""
    rng = np.random.default_rng(seed)
    perc_cov = bca_cov = 0
    widths = []
    for _ in range(R):
        x = rng.exponential(scale=1.0, size=n)   # Exp(1)，真中位数 ln2
        _, boot = bootstrap_medians(x, B, rng)
        plo, phi = percentile_ci(boot)
        blo, bhi, _, _ = bca_ci(x, boot)
        perc_cov += (plo <= true_median <= phi)
        bca_cov += (blo <= true_median <= bhi)
        widths.append(phi - plo)
    p_perc, p_bca = perc_cov / R, bca_cov / R
    mcse = lambda p: float(np.sqrt(p * (1 - p) / R))   # 按观测 p̂ 计，非 p=0.5 上界
    return {"n": n, "B": B, "R": R,
            "perc_coverage": p_perc, "bca_coverage": p_bca,
            "mc_se_perc": mcse(p_perc), "mc_se_bca": mcse(p_bca),
            "mean_perc_width": float(np.mean(widths))}
```

**Step 4: 运行确认通过** → PASS。

**Step 5: Commit**

```bash
git add task3/bootstrap_core.py task3/tests/test_bootstrap_core.py
git commit -m "feat(task3): coverage_experiment with MC SE on observed p-hat"
```

---

## Task 6: 三因素实验 + 主分析运行器

**Files:** Create `task3/experiments.py`

**Step 1: 写运行器（产出数值结果，供图与报告引用）**

```python
"""三因素实验（设计 §4.7）+ 主分析。产出 JSON 供 plot_matplotlib 与报告引用。"""
import json, numpy as np
from task3.bootstrap_core import (load_aircondit7, bootstrap_medians,
                                  percentile_ci, bca_ci, coverage_experiment)

def factor_n_scan(ns=(12, 24, 48, 96), B=2000, R=500, seed=42):
    """因素1：样本量 n（Exp(1) 上抽新鲜样本）。报告 perc/BCa 覆盖率与 CI 宽。"""
    out = []
    for n in ns:
        out.append(coverage_experiment(n=n, B=B, R=R, seed=seed))
    return out

def factor_B_scan(Bs=(100, 1000, 10000), seed=42):
    """因素2：重采样次数 B（aircondit7 上）。报告 percentile/BCa 端点随 B 收敛。"""
    x = load_aircondit7()
    out = []
    for B in Bs:
        rng = np.random.default_rng(seed)
        m_hat, boot = bootstrap_medians(x, B, rng)
        plo, phi = percentile_ci(boot)
        blo, bhi, _, _ = bca_ci(x, boot)
        out.append({"B": B, "perc": [plo, phi], "bca": [blo, bhi], "m_hat": m_hat})
    return out

def factor_outlier_perturbation(ks=(0, 1, 2, 4), extreme=1000.0, B=5000, seed=42):
    """因素3：数据波动——注入 k 个极端值，对比中位数 Bootstrap CI 与均值 t 区间宽度。"""
    from scipy import stats as sps
    x0 = load_aircondit7()
    out = []
    for k in ks:
        x = np.concatenate([x0, np.full(k, extreme)])
        rng = np.random.default_rng(seed)
        _, boot = bootstrap_medians(x, B, rng)
        plo, phi = percentile_ci(boot)
        # 均值 t 区间（参考）
        n = len(x); mean = x.mean(); sd = x.std(ddof=1)
        t = sps.t.ppf(0.975, n - 1)
        t_lo, t_hi = mean - t * sd / np.sqrt(n), mean + t * sd / np.sqrt(n)
        out.append({"k": k, "median_ci_width": phi - plo,
                    "mean_t_width": t_hi - t_lo})
    return out

def main_analysis():
    """主分析：aircondit7 的点估计/SE/percentile/BCa + Exp(1) 主覆盖率。"""
    x = load_aircondit7()
    rng = np.random.default_rng(42)
    m_hat, boot = bootstrap_medians(x, B=10000, rng=rng)
    se = float(np.std(boot, ddof=1))
    plo, phi = percentile_ci(boot)
    blo, bhi, z0, a_hat = bca_ci(x, boot)
    cov = coverage_experiment(n=24, B=2000, R=1000, seed=42)
    return {"m_hat": m_hat, "se_boot": se,
            "percentile_ci": [plo, phi], "bca_ci": [blo, bhi],
            "z0": z0, "a_hat": a_hat,
            "exp1_coverage": cov}

def run_all_experiments(path="task3/output/experiments.json"):
    res = {"main": main_analysis(),
           "factor_n": factor_n_scan(),
           "factor_B": factor_B_scan(),
           "factor_outlier": factor_outlier_perturbation()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("wrote", path)
    return res

if __name__ == "__main__":
    run_all_experiments()
```

**Step 2: 运行并人工核验输出**

Run: `python -m task3.experiments`
Expected: 写出 `task3/output/experiments.json`。
核验：`main.m_hat`≈41.5；`main.percentile_ci` 落在 [20,80] 量级；`main.exp1_coverage.perc_coverage`≈0.90–0.96；`main.bca_ci` 与 `percentile_ci` 在 n=24 应接近（§4.4 退化）；`factor_outlier` 中 `k`↑ 时 `mean_t_width` 暴涨而 `median_ci_width` 几乎不动。

**Step 3: Commit**

```bash
git add task3/experiments.py task3/output/experiments.json
git commit -m "feat(task3): three-factor experiments + main analysis runner"
```

---

## Task 7: 报告图（plot_matplotlib.py，5 张，DPI=300）

**Files:** Create `task3/plot_matplotlib.py`

每图三件套：编号 图3-N / 中英标题 / 图注。统一 `import viz_style; viz_style.apply()`。

**Step 1: 写绘图模块**

```python
"""任务三报告图（设计 §五）。统一 matplotlib + viz_style，DPI=300。"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as sps
import viz_style; viz_style.apply()
from task3.bootstrap_core import (load_aircondit7, bootstrap_medians,
                                  percentile_ci, bca_ci, coverage_experiment)

OUT = "task3/output"

def fig_3_1_sample_distribution():
    x = load_aircondit7()
    fig, ax = plt.subplots()
    ax.hist(x, bins=12, density=True, alpha=0.6, label="原始样本 / raw")
    xs = np.linspace(x.min(), x.max(), 200)
    kde = sps.gaussian_kde(x)
    ax.plot(xs, kde(xs), label="KDE")
    ax.axvline(np.median(x), color="C1", ls="--", label=f"$\\hat m$={np.median(x):.1f}")
    ax.axvline(x.mean(), color="C2", ls=":", label=f"$\\bar x$={x.mean():.1f}")
    ax.set_title("图3-1 原始样本分布 / Raw sample distribution (n=24)")
    ax.set_xlabel("故障间隔 (小时)"); ax.set_ylabel("密度"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/fig3_1_sample.png"); plt.close(fig)

def fig_3_2_bootstrap_evolution():
    x = load_aircondit7()
    fig, ax = plt.subplots()
    for B in (100, 1000, 10000):
        rng = np.random.default_rng(42)
        _, boot = bootstrap_medians(x, B, rng)
        kde = sps.gaussian_kde(boot)
        xs = np.linspace(boot.min(), boot.max(), 200)
        ax.plot(xs, kde(xs), label=f"B={B}")
    ax.set_title("图3-2 Bootstrap 中位数分布演化 / Bootstrap median distribution (B↑)")
    ax.set_xlabel("中位数 (小时)"); ax.set_ylabel("密度"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/fig3_2_evolution.png"); plt.close(fig)

def fig_3_3_endpoint_stability():
    x = load_aircondit7()
    Bs = np.unique(np.round(np.logspace(2, 4, 12)).astype(int))
    perc_lo, perc_hi, bca_lo, bca_hi = [], [], [], []
    for B in Bs:
        rng = np.random.default_rng(42)
        _, boot = bootstrap_medians(x, B, rng)
        plo, phi = percentile_ci(boot)
        blo, bhi, _, _ = bca_ci(x, boot)
        perc_lo.append(plo); perc_hi.append(phi); bca_lo.append(blo); bca_hi.append(bhi)
    fig, ax = plt.subplots()
    ax.plot(Bs, perc_lo, "o-", label="percentile 下端")
    ax.plot(Bs, perc_hi, "o-", label="percentile 上端")
    ax.set_xscale("log")
    ax.set_title("图3-3 CI 端点随 B 收敛 / Endpoint stability vs B")
    ax.set_xlabel("B (重采样次数)"); ax.set_ylabel("端点 (小时)"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/fig3_3_stability.png"); plt.close(fig)

def fig_3_4_ci_comparison():
    x = load_aircondit7()
    rng = np.random.default_rng(42)
    m_hat, boot = bootstrap_medians(x, 10000, rng)
    plo, phi = percentile_ci(boot)
    blo, bhi, _, _ = bca_ci(x, boot)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.errorbar([1], [m_hat], yerr=[[m_hat-plo], [phi-m_hat]], fmt="o", capsize=8, label="percentile")
    ax.errorbar([2], [m_hat], yerr=[[m_hat-blo], [bhi-m_hat]], fmt="o", capsize=8, label="BCa")
    ax.set_xticks([1, 2]); ax.set_xticklabels(["percentile", "BCa"])
    ax.set_title("图3-4 percentile vs BCa CI / CI comparison (n=24)")
    ax.set_ylabel("故障间隔 (小时)"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/fig3_4_ci.png"); plt.close(fig)

def fig_3_5_three_factor():
    # (a) CI 宽 vs n + 1/sqrt(n) 拟合； (b) 覆盖率柱状图 ±MC SE； (c) 离群点扰动
    ns = (12, 24, 48, 96)
    cov = [coverage_experiment(n=n, B=2000, R=300, seed=42) for n in ns]
    widths = [c["mean_perc_width"] for c in cov]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    # (a)
    axes[0].plot(ns, widths, "o-", label="实测 CI 宽")
    fit = [widths[1] * np.sqrt(ns[1]) / np.sqrt(n) for n in ns]
    axes[0].plot(ns, fit, "--", label="$1/\\sqrt{n}$ 拟合")
    axes[0].set_title("(a) CI 宽 vs n"); axes[0].set_xlabel("n"); axes[0].legend()
    # (b)
    pp = [c["perc_coverage"] for c in cov]; pb = [c["bca_coverage"] for c in cov]
    ep = [c["mc_se_perc"] for c in cov]; eb = [c["mc_se_bca"] for c in cov]
    xb = np.arange(len(ns))
    axes[1].bar(xb-0.2, pp, 0.4, yerr=ep, label="percentile")
    axes[1].bar(xb+0.2, pb, 0.4, yerr=eb, label="BCa")
    axes[1].axhline(0.95, color="gray", ls=":")
    axes[1].set_xticks(xb); axes[1].set_xticklabels(ns)
    axes[1].set_title("(b) 覆盖率 ± MC SE"); axes[1].set_xlabel("n"); axes[1].set_ylim(0.8,1.0); axes[1].legend()
    # (c)
    from task3.experiments import factor_outlier_perturbation
    fo = factor_outlier_perturbation()
    ks = [d["k"] for d in fo]; mw = [d["median_ci_width"] for d in fo]; tw = [d["mean_t_width"] for d in fo]
    axes[2].plot(ks, mw, "o-", label="中位数 Bootstrap CI")
    axes[2].plot(ks, tw, "s-", label="均值 t 区间")
    axes[2].set_title("(c) 离群点扰动"); axes[2].set_xlabel("注入极端值数 k"); axes[2].legend()
    fig.suptitle("图3-5 三因素对比 / Three-factor comparison")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig3_5_three_factor.png"); plt.close(fig)

def main():
    fig_3_1_sample_distribution()
    fig_3_2_bootstrap_evolution()
    fig_3_3_endpoint_stability()
    fig_3_4_ci_comparison()
    fig_3_5_three_factor()
    print("wrote fig3_1..fig3_5 to", OUT)

if __name__ == "__main__":
    main()
```

**Step 2: 生成全部图**

Run: `python -m task3.plot_matplotlib`
Expected: `task3/output/` 下生成 5 个 PNG（fig3_1..fig3_5），DPI=300。

**Step 3: 人工目检（每图三件套是否齐）** — 打开 5 张图确认：中文不豆腐块、图注清晰、(a)/(b)/(c) 子图可读、BCa 与 percentile 在 n=24 图中接近（§4.4）。

**Step 4: Commit**

```bash
git add task3/plot_matplotlib.py task3/output/*.png
git commit -m "feat(task3): five report figures (structure/process/result), DPI=300"
```

---

## Task 8: run_all.py 一键复现 + README

**Files:** Create `task3/run_all.py`, `task3/README.md`

**Step 1: 写 run_all.py**

```python
"""一键复现：跑测试 → 跑实验 → 出图。命令：python -m task3.run_all"""
import subprocess, sys

def main():
    subprocess.run([sys.executable, "-m", "pytest", "task3/tests/", "-v"], check=False)
    from task3.experiments import run_all_experiments
    run_all_experiments()
    from task3 import plot_matplotlib
    plot_matplotlib.main()
    print("\nDone. 见 task3/output/ 与 experiments.json")

if __name__ == "__main__":
    main()
```

**Step 2: 写 README.md（含运行说明 + 引用 + AI 声明）**

```markdown
# 任务三：Bootstrap 估计中位数

## 运行
python -m pip install -r requirements.txt
python -m task3.run_all          # 测试 + 实验 + 出图，一键复现
streamlit run task3/app.py       # 交互仪表盘（可选）

## 数据
`task3/data/aircondit7.csv`：Proschan (1963) 波音720第7架机空调故障间隔，n=24，单位服务小时。
来源：R `boot` 包 `aircondit7`；原始论文 Proschan, F. (1963), Technometrics 5(3):375–383。

## 产物
- `task3/output/experiments.json`：全部数值结果
- `task3/output/fig3_1..fig3_5.png`：报告图（DPI=300）

## AI 使用声明
计算核心与图表代码、公式符号核验借助 AI 辅助；**报告中的分析结论（结果分析、综合讨论、局限性反思）由小组自行撰写**。
```

**Step 3: 跑一次 run_all 确认端到端**

Run: `python -m task3.run_all`
Expected: 测试 PASS、写出 experiments.json + 5 图，无报错。

**Step 4: Commit**

```bash
git add task3/run_all.py task3/README.md
git commit -m "feat(task3): one-command run_all + README with citation & AI note"
```

---

## Task 9（可选·加分）: Streamlit 交互仪表盘

**Files:** Create `task3/app.py`

**Step 1: 写仪表盘**（设计 §6.3，复用 bootstrap_core）

```python
"""Streamlit 仪表盘（加分项）。复用 bootstrap_core，不改报告图。运行：streamlit run task3/app.py"""
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import viz_style; viz_style.apply()
from task3.bootstrap_core import (load_aircondit7, bootstrap_medians,
                                  percentile_ci, bca_ci)

st.set_page_config(page_title="Bootstrap 中位数", layout="wide")
st.sidebar.header("参数")
data_src = st.sidebar.radio("数据", ["aircondit7", "Exp(1) 模拟"])
n = st.sidebar.slider("样本量 n", 12, 200, 24, 2)
B = st.sidebar.slider("重采样次数 B", 100, 20000, 10000, 100)
seed = st.sidebar.number_input("随机种子", 0, 9999, 42)
run = st.sidebar.button("重新重采样")

rng = np.random.default_rng(int(seed))
if data_src == "aircondit7":
    x = load_aircondit7()[:n] if n <= 24 else np.resize(load_aircondit7(), n)
else:
    x = rng.exponential(size=n)
m_hat, boot = bootstrap_medians(x, B, rng)
plo, phi = percentile_ci(boot)
blo, bhi, z0, a_hat = bca_ci(x, boot)

tab1, tab2, tab3 = st.tabs(["原始+Bootstrap分布", "CI 对比", "说明"])
with tab1:
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(); ax.hist(x, bins=15, alpha=0.6)
        ax.axvline(np.median(x), color="C1", ls="--")
        st.pyplot(fig)
    with c2:
        fig2, ax2 = plt.subplots(); ax2.hist(boot, bins=40, alpha=0.6)
        ax2.axvline(plo, color="r"); ax2.axvline(phi, color="r")
        ax2.axvline(blo, color="g", ls="--"); ax2.axvline(bhi, color="g", ls="--")
        st.pyplot(fig2)
with tab2:
    st.write(f"点估计 $\\hat m$ = **{m_hat:.3f}**")
    st.write(f"percentile 95% CI = [{plo:.2f}, {phi:.2f}]")
    st.write(f"BCa 95% CI = [{blo:.2f}, {bhi:.2f}]  (z0={z0:.3f}, a_hat={a_hat:.4f})")
    st.caption("注：中位数偶数 n 下 a_hat≈0，BCa≈percentile（设计 §4.4）。")
with tab3:
    st.markdown("方法/假设/为何用 Bootstrap 见《任务三设计.md》。"
                "仪表盘为加分项，报告图仍用 matplotlib（plot_matplotlib.py）。")
```

**Step 2: 本地启动确认**

Run: `streamlit run task3/app.py`（手动浏览器打开，拖滑块看交互；Ctrl+C 关闭）。
Expected: 页面正常，拖 n/B、按按钮实时重采样。

**Step 3: Commit**

```bash
git add task3/app.py
git commit -m "feat(task3): optional Streamlit dashboard (additive, reuses core)"
```

---

## 完成判据（DoD）

- [ ] `python -m pytest task3/tests/` 全绿（含 BCa 中位数退化测试）
- [ ] `python -m task3.run_all` 一键产出 experiments.json + 5 图，无报错
- [ ] experiments.json 数值核验通过：m_hat≈41.5、Exp(1) n=24 percentile 覆盖≈0.90–0.96（报 ±MC SE）、离群点下均值 t 宽暴涨而中位数 CI 稳
- [ ] 5 张图每张三件套齐全（编号/中英标题/图注）、中文不豆腐块、DPI=300
- [ ] README 含运行说明 + Proschan/R boot 引用 + AI 使用声明
- [ ] （可选）app.py 可启动交互

## 提交后回到报告

数值与图齐备后，回填 [任务三设计.md](../../任务三设计.md) §4.5/§7.4 中标注"待实测回填"的覆盖率数值（用 experiments.json 的真实结果），再据此撰写实践报告"任务三"章节（模型层/算法层/图/分析）。
