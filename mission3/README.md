# 任务三：Bootstrap 估计中位数

> 课程：数学思维实践（CST4822A）· 任务三 · 概率论与数理统计实验（满分 15）

对 Proschan (1963) 波音 720 空调故障间隔数据（`aircondit7`，n=24），用**非参数 Bootstrap** 估计总体中位数，给出 percentile + BCa 置信区间，用 Exp(1) 模拟验证覆盖率，并做样本量 / 重采样次数 / 数据波动三因素实验。

## 快速开始

依赖：Python 3.10+，见仓库根 [`requirements.txt`](../requirements.txt)（`numpy<2 / scipy / matplotlib / pytest / streamlit`）。

```bash
# 安装依赖（从仓库根）
pip install -r requirements.txt

# 一键复现：跑测试 → 跑实验 → 出图（约 30 秒）
python -m mission3.run_all                  # 从仓库根
python -m run_all                        # 或从 mission3/ 内（两者等价）

# 交互仪表盘（可选）
python -m streamlit run mission3/app.py    # 从仓库根
python -m streamlit run app.py          # 或从 mission3/ 内（若 streamlit 已在 PATH，裸命令 `streamlit run ...` 亦可）
```

> 路径已做 **cwd 无关**处理，从仓库根或 `mission3/` 内运行均可。⚠️ **不要**在 `mission3/` 内用 `python -m mission3.run_all`——Python 包机制限制会报 `No module named 'mission3'`；从 `mission3/` 内请用 `python -m run_all`。

## 数据

`mission3/data/aircondit7.csv` — Proschan (1963) 第 7 架波音 720 空调故障间隔，**n=24**，单位服务小时。

| 统计量 | 值 |
|---|---|
| 中位数 | 41.5 h |
| 均值 | 64.12 h |
| 标准差 | 62.65 h |
| 极差 | [3, 210] |
| 均值 / 中位数 | 1.55（明显右偏） |

**来源与可复现**：R `boot` 包 `aircondit7`（`library(boot); data(aircondit7)` 一行加载）；原始论文 Proschan, F. (1963), *Technometrics* 5(3):375–383。

## 主要结果

`run_all` 后全部数值写入 `output/experiments.json`，要点：

**点估计与区间**（aircondit7，B=10000）：
- 中位数 $\hat m$ = **41.5** h，Bootstrap 标准误 SE = **13.89** h
- percentile 95% CI = **[22.0, 79.0]**
- BCa 95% CI = **[22.0, 75.5]**（z₀ = −0.087，â = **0.0**）

**Exp(1) 覆盖率验证**（n=24，B=2000，R=1000，真中位数 ln2≈0.693）：
- percentile 覆盖率 = **0.934 ± 0.0079**
- BCa 覆盖率 = **0.928 ± 0.0082**
- 解析标尺：SE ≈ 1/√24 ≈ 0.204，95% 区间宽 ≈ 0.80（实测 0.796，吻合）

**三因素实验**：
- 样本量 $n$↑（Exp(1)，R=500）：CI 宽 1.16 → 0.40（n=12 → 96），近似按 $1/\sqrt n$ 收窄
- 重采样次数 $B$↑：CI 端点逐步稳定
- 数据波动（注入离群点 k=0 → 4）：中位数 CI 宽 57 → 73（稳），均值 t 区间宽 52.9 → 262.5（被拉宽 ~5×）→ 中位数稳健

## 方法要点（写报告时注意）

- **为何选中位数 + Bootstrap**：中位数的渐近 SE $1/(2f(m)\sqrt n)$ 依赖未知密度 $f(m)$，无简单闭式 CI；而均值 / 比例 / 两样本均值差都有 t / Wilson / Welch 区间。Bootstrap 恰补此空白——这正是任务书"Bootstrap 适合该问题的原因"评分点。
- **BCa 在中位数下退化（关键）**：中位数是非光滑泛函，偶数 $n$ 下留一法中位数塌缩为 2 个值 → jackknife 加速度 $\hat a\equiv 0$（实测精确为 0）。故 **BCa 退化为 percentile + 偏差修正**，在所有偶数 $n$ 都与 percentile 近乎重合、**不修正欠覆盖**。$n$ 扫描（12/24/48/96）下 percentile / BCa 覆盖率始终在一倍 MC SE 内。
- **轻度欠覆盖是发现不是 bug**：percentile 法一阶精度 $O(n^{-1/2})$ 在偏态数据下约 0.93 的欠覆盖是已知性质；MC SE 一律按观测 $\hat p$ 计（√($\hat p(1-\hat p)/R$)，非 p=0.5 上界）。

## 项目结构

```
mission3/
  README.md              # 本文件
  run_all.py             # 一键复现入口
  bootstrap_core.py      # 计算核心：load_aircondit7 / bootstrap_medians / percentile_ci / bca_ci / coverage_experiment
  experiments.py         # 主分析 + 三因素实验 → experiments.json
  plot_matplotlib.py     # 5 张报告图（DPI=300）
  app.py                 # Streamlit 交互仪表盘（加分项）
  viz_style.py           # 任务三自包含绘图风格（不与任务一/二共享）
  data/aircondit7.csv    # Proschan 数据 n=24
  tests/test_bootstrap_core.py  # 8 个单元测试（含 BCa 中位数退化、scipy 交叉验证）
  output/                # 生成产物（已 gitignore，run_all 重新生成）
    experiments.json
    fig3_1..fig3_5.png
```

## 产物

`run_all` 生成于 `output/`（已 `.gitignore`，不进版本库，随时重新生成）：

| 文件 | 内容 |
|---|---|
| `experiments.json` | 全部数值结果 |
| `fig3_1_sample.png` | 原始样本分布（直方图 + KDE + 箱线图） |
| `fig3_2_evolution.png` | Bootstrap 中位数分布随 B 演化 |
| `fig3_3_stability.png` | percentile / BCa CI 端点随 B 收敛 |
| `fig3_4_ci.png` | percentile vs BCa CI 对比 |
| `fig3_5_three_factor.png` | 三因素对比（CI 宽 / 覆盖率 / 离群点扰动） |

每张图三件套：编号（图3-N）/ 中英标题 / 轴标签，DPI=300，中文字体无豆腐块。

## 可复现性

固定随机种子（`np.random.default_rng(42)`）：`experiments.json` 数值结果可逐字复现；5 张 PNG 在相同 Python / matplotlib / 字体环境下亦可字节级复现（已验证 md5 不变）。`python -m pytest mission3/tests/` 8/8 全绿。

## AI 使用声明

计算核心与图表代码、公式符号核验、数据真伪核验借助 AI 辅助；**报告中的分析结论（结果分析、综合讨论、局限性反思）由小组自行撰写**，不由 AI 代写。

## 引用

- Proschan, F. (1963). *Theoretical Explanation of Observed Decreasing Failure Rate*. **Technometrics**, 5(3), 375–383. DOI: 10.1080/00401706.1963.10490105
- R `boot` 包数据集 `aircondit7`：https://github.com/cran/boot
- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.（percentile eq 13.5；BCa eq 14.10/14.15）
- Davison, A. C., & Hinkley, D. V. (1997). *Bootstrap Methods and Their Application*. Cambridge University Press.
