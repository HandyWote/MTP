"""三因素实验（设计 §4.7）+ 主分析。产出 JSON 供 plot_matplotlib 与报告引用。"""
import json
from pathlib import Path
import numpy as np
from mission3.bootstrap_core import (load_aircondit7, bootstrap_medians,
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


def run_all_experiments(path=None):
    if path is None:
        path = Path(__file__).resolve().parent / "output" / "experiments.json"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)   # fresh-clone 安全：output/ 被 .gitignore，须由代码自建
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
