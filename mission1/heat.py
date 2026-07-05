"""任务一：迭代法解稳态热传导 Ax=b。单文件函数化。"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti TC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300


def build_1d(n, T_left, T_right):
    """铁棍：n 个内部点，两端温度固定。三对角 A，2 在主对角、-1 在紧邻。"""
    A = np.zeros((n, n))
    b = np.zeros(n)
    for i in range(n):
        A[i, i] = 2
        if i > 0: A[i, i-1] = -1
        if i < n-1: A[i, i+1] = -1
    b[0] += T_left
    b[-1] += T_right
    return A, b


def build_2d(n, T_top, T_others):
    """铁板 n×n 内部点，五点差分。上边=T_top，左/右/下=T_others。
    方程：4·T_ij - T_上 - T_下 - T_左 - T_右 = b_ij；边界邻居移到 b。"""
    N = n * n
    A = np.zeros((N, N))
    b = np.zeros(N)
    def idx(i, j): return i * n + j
    for i in range(n):
        for j in range(n):
            k = idx(i, j)
            A[k, k] = 4
            # 上邻
            if i > 0: A[k, idx(i-1, j)] = -1
            else: b[k] += T_top
            # 下邻
            if i < n-1: A[k, idx(i+1, j)] = -1
            else: b[k] += T_others
            # 左邻
            if j > 0: A[k, idx(i, j-1)] = -1
            else: b[k] += T_others
            # 右邻
            if j < n-1: A[k, idx(i, j+1)] = -1
            else: b[k] += T_others
    return A, b


def solve_direct(A, b):
    """直接法（裁判）。"""
    return np.linalg.solve(A, b)


def jacobi(A, b, x0, tol=1e-6, max_iter=10000, snapshot_every=None):
    """Jacobi：整轮用旧值（双缓冲 x / x_new）。"""
    n = len(b)
    x = np.asarray(x0, dtype=float).copy()
    x_new = x.copy()
    err_hist, snaps = [], []
    if snapshot_every: snaps.append(x.copy())  # 初值快照
    for it in range(max_iter):
        for i in range(n):
            s = b[i] - A[i] @ x + A[i, i] * x[i]   # 减去 A[i,i]*x[i] 因含在对角
            x_new[i] = s / A[i, i]
        diff = np.max(np.abs(x_new - x))
        x, x_new = x_new, x        # 换引用（旧 x 的内存复用为下一轮 x_new）
        err_hist.append(diff)
        if snapshot_every and (it + 1) % snapshot_every == 0:
            snaps.append(x.copy())
        if diff < tol:
            break
    return x, err_hist, snaps


def gauss_seidel(A, b, x0, tol=1e-6, max_iter=10000, snapshot_every=None):
    """Gauss-Seidel：算过的分量立即用新值（in-place 就地更新）。"""
    n = len(b)
    x = np.asarray(x0, dtype=float).copy()
    err_hist, snaps = [], []
    if snapshot_every: snaps.append(x.copy())
    for it in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            s = b[i] - A[i] @ x + A[i, i] * x[i]
            x[i] = s / A[i, i]
        diff = np.max(np.abs(x - x_old))
        err_hist.append(diff)
        if snapshot_every and (it + 1) % snapshot_every == 0:
            snaps.append(x.copy())
        if diff < tol:
            break
    return x, err_hist, snaps
