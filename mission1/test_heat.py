import numpy as np
from numpy.testing import assert_allclose
from heat import build_1d, build_2d, solve_direct
from heat import jacobi, gauss_seidel

def test_build_2d_2x2():
    A, b = build_2d(2, 100, 0)
    A_exp = np.array([[4,-1,-1,0],[-1,4,0,-1],[-1,0,4,-1],[0,-1,-1,4]], dtype=float)
    assert_allclose(A, A_exp)
    assert_allclose(b, [100,100,0,0])

def test_solve_2x2():
    A, b = build_2d(2, 100, 0)
    assert_allclose(solve_direct(A, b), [37.5,37.5,12.5,12.5], atol=1e-9)

def test_build_1d_and_solve():
    A, b = build_1d(5, 100, 0)
    assert A.shape == (5,5) and b.shape == (5,)
    for i in range(5):
        for j in range(5):
            if abs(i-j) > 1: assert A[i,j] == 0  # 三对角
    x = solve_direct(A, b)
    assert_allclose(x, [100*(1-k/6) for k in range(1,6)], atol=1e-9)

def test_jacobi_2x2():
    A, b = build_2d(2, 100, 0)
    x, err, snaps = jacobi(A, b, np.zeros(4))
    assert_allclose(x, [37.5,37.5,12.5,12.5], atol=1e-6)
    assert len(snaps) == 0 and len(err) > 0

def test_iter_converge_30_and_gs_faster():
    A, b = build_2d(30, 100, 0)
    xd = solve_direct(A, b)
    xj, ej, sj = jacobi(A, b, np.zeros(900), snapshot_every=50)
    xg, eg, sg = gauss_seidel(A, b, np.zeros(900), snapshot_every=50)
    assert_allclose(xj, xd, atol=5e-4)   # 迭代停止 tol=1e-6 → 真解偏差 ~1.9e-4，留余量
    assert_allclose(xg, xd, atol=5e-4)
    assert len(eg) < len(ej)          # GS 收敛更快
    assert len(sj) > 0 and len(sg) > 0

def _run_all():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"): fn(); print(f"  ✓ {name}")

if __name__ == "__main__":
    _run_all(); print("Task1 OK")
