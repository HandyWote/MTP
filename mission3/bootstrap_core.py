import csv
from pathlib import Path
import numpy as np
from scipy import stats as sps


def load_aircondit7(path=None):
    """Proschan (1963) 第7架机空调故障间隔（小时），n=24。"""
    if path is None:
        path = Path(__file__).resolve().parent / "data" / "aircondit7.csv"
    vals = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vals.append(float(row["hours"]))
    return np.array(vals, dtype=float)


def bootstrap_medians(x, B, rng):
    """非参数有放回重采样 B 次，每次算中位数。
    返回 (m_hat, boot_medians)。向量化：一次性抽 B×n 索引。"""
    x = np.asarray(x, dtype=float)
    n = len(x)
    idx = rng.integers(0, n, size=(B, n))   # 有放回（允许重复索引）
    boot = np.median(x[idx], axis=1)
    return float(np.median(x)), boot


def percentile_ci(boot, alpha=0.05):
    """percentile 法：取 bootstrap 分布的 α/2 与 1−α/2 分位。"""
    lo = float(np.quantile(boot, alpha / 2))
    hi = float(np.quantile(boot, 1 - alpha / 2))
    return lo, hi


def bca_ci(x, boot, alpha=0.05):
    """BCa：偏差修正 z0 + jackknife 加速度 a_hat。返回 (lo, hi, z0, a_hat)。设计 §4.4。"""
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
