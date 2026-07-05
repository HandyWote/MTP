"""任务三报告图（设计 §五）。matplotlib + 任务三自包含风格 mission3.viz_style，DPI=300。"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as sps
from pathlib import Path
from mission3 import viz_style; viz_style.apply()
from mission3.bootstrap_core import (load_aircondit7, bootstrap_medians,
                                  percentile_ci, bca_ci)

OUT = str(Path(__file__).resolve().parent / "output")


def fig_3_1_sample_distribution():
    x = load_aircondit7()
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7, 5),
                                  gridspec_kw={"height_ratios": [4, 0.5]}, sharex=True)
    ax.hist(x, bins=12, density=True, alpha=0.6, label="原始样本 / raw")
    xs = np.linspace(x.min(), x.max(), 200)
    kde = sps.gaussian_kde(x)
    ax.plot(xs, kde(xs), label="KDE")
    ax.axvline(np.median(x), color="black", ls="--", label=f"$\\hat m$={np.median(x):.1f}")
    ax.axvline(x.mean(), color="C2", ls=":", label=f"$\\bar x$={x.mean():.1f}")
    ax.set_title("图3-1 原始样本分布 / Raw sample distribution (n=24)")
    ax.set_ylabel("密度"); ax.legend()
    ax2.boxplot(x, vert=False, widths=0.6)
    ax2.set_yticks([]); ax2.set_xlabel("故障间隔 (小时)")
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
    ax.plot(Bs, bca_lo, "s--", label="BCa 下端", alpha=0.6)
    ax.plot(Bs, bca_hi, "s--", label="BCa 上端", alpha=0.6)
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
    ax.set_ylabel("故障间隔 (小时)"); ax.set_xlabel("方法"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/fig3_4_ci.png"); plt.close(fig)


def fig_3_5_three_factor():
    import json, os
    path = f"{OUT}/experiments.json"
    if not os.path.exists(path):              # standalone-safety: generate if missing
        from mission3.experiments import run_all_experiments
        run_all_experiments(path)
    with open(path, encoding="utf-8") as f:
        res = json.load(f)
    factor_n = res["factor_n"]                # R=500, consistent with experiments.json
    factor_outlier = res["factor_outlier"]
    ns = [d["n"] for d in factor_n]
    widths = [d["mean_perc_width"] for d in factor_n]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    # (a) CI 宽 vs n + 1/sqrt(n) 拟合
    axes[0].plot(ns, widths, "o-", label="实测 CI 宽")
    fit = [widths[1] * np.sqrt(ns[1]) / np.sqrt(n) for n in ns]
    axes[0].plot(ns, fit, "--", label="$1/\\sqrt{n}$ 拟合")
    axes[0].set_title("(a) CI 宽 vs n"); axes[0].set_xlabel("n"); axes[0].set_ylabel("CI 宽度 (小时)"); axes[0].legend()
    # (b) 覆盖率柱状图 ± MC SE
    pp = [d["perc_coverage"] for d in factor_n]; pb = [d["bca_coverage"] for d in factor_n]
    ep = [d["mc_se_perc"] for d in factor_n]; eb = [d["mc_se_bca"] for d in factor_n]
    xb = np.arange(len(ns))
    axes[1].bar(xb-0.2, pp, 0.4, yerr=ep, label="percentile")
    axes[1].bar(xb+0.2, pb, 0.4, yerr=eb, label="BCa")
    axes[1].axhline(0.95, color="gray", ls=":")
    axes[1].set_xticks(xb); axes[1].set_xticklabels(ns)
    axes[1].set_title("(b) 覆盖率 ± MC SE"); axes[1].set_xlabel("n"); axes[1].set_ylabel("覆盖率"); axes[1].set_ylim(0.8, 1.0); axes[1].legend()
    # (c) 离群点扰动
    ks = [d["k"] for d in factor_outlier]; mw = [d["median_ci_width"] for d in factor_outlier]; tw = [d["mean_t_width"] for d in factor_outlier]
    axes[2].plot(ks, mw, "o-", label="中位数 Bootstrap CI")
    axes[2].plot(ks, tw, "s-", label="均值 t 区间")
    axes[2].set_title("(c) 离群点扰动"); axes[2].set_xlabel("注入极端值数 k"); axes[2].set_ylabel("区间宽度 (小时)"); axes[2].legend()
    fig.suptitle("图3-5 三因素对比 / Three-factor comparison")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig3_5_three_factor.png"); plt.close(fig)


def main():
    Path(OUT).mkdir(parents=True, exist_ok=True)   # fresh-clone 安全：output/ 被 .gitignore，须由代码自建
    fig_3_1_sample_distribution()
    fig_3_2_bootstrap_evolution()
    fig_3_3_endpoint_stability()
    fig_3_4_ci_comparison()
    fig_3_5_three_factor()
    print("wrote fig3_1..fig3_5 to", OUT)


if __name__ == "__main__":
    main()
