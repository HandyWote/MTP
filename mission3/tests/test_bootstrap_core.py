import numpy as np
from mission3.bootstrap_core import (
    load_aircondit7,
    bootstrap_medians,
    percentile_ci,
    bca_ci,
    coverage_experiment,
)

KNOWN = [3,5,5,13,14,15,22,22,23,30,36,39,44,46,50,
         72,79,88,97,102,139,188,197,210]  # list（非 set）—保留重复值

def test_load_aircondit7():
    x = load_aircondit7()
    assert len(x) == 24
    assert np.median(x) == 41.5
    assert sorted(x.tolist()) == sorted(KNOWN)


def test_bootstrap_medians_reproducible_and_correct():
    x = load_aircondit7()
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    m1, b1 = bootstrap_medians(x, B=1000, rng=rng1)
    m2, b2 = bootstrap_medians(x, B=1000, rng=rng2)
    assert m1 == 41.5                      # 点估计 = 原样本中位数
    assert len(b1) == 1000
    assert np.array_equal(b1, b2)          # 固定种子可复现
    assert b1.min() >= x.min() and b1.max() <= x.max()  # 重采样中位数落在样本范围内


def test_percentile_ci_known_and_contains_estimate():
    arr = np.arange(1, 101)
    assert percentile_ci(arr, alpha=0.05) == (3.475, 97.52499999999999)
    x = load_aircondit7()
    rng = np.random.default_rng(0)
    m_hat, boot = bootstrap_medians(x, 5000, rng)
    lo, hi = percentile_ci(boot)
    assert lo <= m_hat <= hi


def test_bca_degenerates_to_percentile_for_symmetric():
    rng = np.random.default_rng(1)
    x = rng.normal(size=200)               # 对称数据
    m_hat, boot = bootstrap_medians(x, 4000, rng)
    plo, phi = percentile_ci(boot)
    blo, bhi, z0, a_hat = bca_ci(x, boot)
    assert abs(z0) < 0.1
    assert abs(blo - plo) < abs(phi - plo) * 0.1 + 1e-9
    assert abs(bhi - phi) < abs(phi - plo) * 0.1 + 1e-9


def test_bca_median_even_n_a_hat_near_zero():
    # 设计 §4.4 核心退化：中位数 + 偶数 n → jackknife a_hat ≈ 0（留一法中位数塌缩为 2 值，对称分裂→三次方和恰为 0）
    x = load_aircondit7()                  # n=24 偶数
    rng = np.random.default_rng(2)
    _, boot = bootstrap_medians(x, 4000, rng)
    _, _, _, a_hat = bca_ci(x, boot)
    assert abs(a_hat) < 1e-6              # 数学上应为精确 0.0


def test_bca_matches_scipy_on_small_case():
    import pytest
    try:
        from scipy import stats as sps
    except ImportError:
        pytest.skip("scipy 不可用")
    x = np.array([1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20])
    rng = np.random.default_rng(3)
    _, boot = bootstrap_medians(x, 2000, rng)
    blo, bhi, _, _ = bca_ci(x, boot)
    res = sps.bootstrap((x,), np.median, method="BCa",
                        n_resamples=2000, random_state=3, vectorized=False)
    assert res.confidence_interval.low < bhi and res.confidence_interval.high > blo


def test_coverage_experiment_exp1():
    res = coverage_experiment(n=24, B=1000, R=300, seed=42)
    assert 0.80 < res["perc_coverage"] < 1.0   # Exp(1) n=24 应在 ~0.93 附近
    assert 0.80 < res["bca_coverage"] < 1.0
    import math
    assert abs(res["mc_se_perc"] - math.sqrt(
        res["perc_coverage"]*(1-res["perc_coverage"])/300)) < 1e-9
    assert res["mean_perc_width"] > 0

def test_coverage_reproducible():
    r1 = coverage_experiment(n=24, B=500, R=100, seed=7)
    r2 = coverage_experiment(n=24, B=500, R=100, seed=7)
    assert r1 == r2
