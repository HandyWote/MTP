"""任务一：迭代法解稳态热传导 Ax=b。单文件函数化。"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

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


# ────────────────────────────────────────────────────────────────────
# 谱半径（迭代收敛速率的解析刻画）
# 模型问题：5 点差分 / 三对角 Jacobi 迭代矩阵谱半径
#   ρ_J(n)  = cos(π/(n+1))         —— Jacobi
#   ρ_GS(n) = ρ_J(n)²              —— Gauss-Seidel（同模型问题）
# 收敛需 ρ < 1；ρ 越接近 1 收敛越慢。
# 例如 n=30：ρ_J = cos(π/31) ≈ 0.9949，ρ_GS ≈ 0.9898。
# ────────────────────────────────────────────────────────────────────
def spectral_radius_jacobi(n):
    """Jacobi 迭代矩阵谱半径：cos(π/(n+1))。"""
    return np.cos(np.pi / (n + 1))


def spectral_radius_gs(n):
    """Gauss-Seidel 迭代矩阵谱半径：ρ_J²。"""
    return spectral_radius_jacobi(n) ** 2


# ────────────────────────────────────────────────────────────────────
# 可视化（6 张静态图）
# 统一规范：fig.savefig(dpi=300, bbox_inches='tight') → plt.close(fig)
# ────────────────────────────────────────────────────────────────────
def plot_rod(x_5, path):
    """铁棍稳态温度：直接解 vs 解析解（n=5 内部点）。"""
    fig, ax = plt.subplots(figsize=(7, 4))
    k = np.arange(1, len(x_5) + 1)
    analytic = 100 * (1 - k / 6)            # 解析解：T = 100·(1 − k/6)
    ax.plot(k, x_5, 'o-', label='直接解')
    ax.plot(k, analytic, 's--', label='解析解')
    ax.set_xlabel('内部点位置 k')
    ax.set_ylabel('温度 (°C)')
    ax.set_title('铁棍稳态温度：直接解 vs 解析解')
    ax.set_xticks(k)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_matrix_and_bars(A_4x4, x_4, path):
    """左：4×4 系数矩阵稀疏结构；右：4 个内部点温度柱状图。"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].spy(A_4x4)
    axes[0].set_title('4×4 矩阵稀疏结构')
    axes[1].bar(np.arange(1, len(x_4) + 1), x_4,
                color=['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4'])
    axes[1].set_xlabel('内部点编号')
    axes[1].set_ylabel('温度 (°C)')
    axes[1].set_title('4 个内部点稳态温度')
    axes[1].set_xticks(np.arange(1, len(x_4) + 1))
    fig.suptitle('铁板 2×2：系数矩阵与稳态温度分布', fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_heatmap_compare(x_iter, x_direct, n, path):
    """30×30 铁板：迭代解 vs 直接解 热力图。色标 RdBu_r，0–100。"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    im0 = axes[0].imshow(x_iter.reshape(n, n),
                         cmap='RdBu_r', vmin=0, vmax=100, origin='upper')
    axes[0].set_title('迭代解')
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    im1 = axes[1].imshow(x_direct.reshape(n, n),
                         cmap='RdBu_r', vmin=0, vmax=100, origin='upper')
    axes[1].set_title('直接解')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.suptitle(f'铁板 {n}×{n} 稳态温度场：迭代解 vs 直接解', fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_error_curve(err_j, err_g, path):
    """Jacobi / GS 迭代误差曲线（半对数）。GS 应位于 Jacobi 之下。"""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.semilogy(np.arange(1, len(err_j) + 1), err_j, '-o', markersize=3,
                label='Jacobi')
    ax.semilogy(np.arange(1, len(err_g) + 1), err_g, '-s', markersize=3,
                label='Gauss-Seidel')
    ax.set_xlabel('迭代轮数')
    ax.set_ylabel('步差 ‖x_新 − x_旧‖∞')
    ax.set_title('迭代误差曲线（半对数）')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_method_bars(stats, path):
    """三方法（Jacobi / GS / 直接法）三指标对比：轮数 / 误差 / 耗时(ms)。"""
    names = list(stats.keys())
    iters = [stats[k][0] for k in names]
    errs = [stats[k][1] for k in names]
    times = [stats[k][2] for k in names]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    x = np.arange(len(names))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    axes[0].bar(x, iters, color=colors)
    axes[0].set_xticks(x); axes[0].set_xticklabels(names, rotation=15)
    axes[0].set_ylabel('迭代轮数')
    axes[0].set_title('收敛轮数')
    axes[1].bar(x, errs, color=colors)
    axes[1].set_xticks(x); axes[1].set_xticklabels(names, rotation=15)
    axes[1].set_ylabel('最终步差')
    axes[1].set_title('收敛误差')
    axes[2].bar(x, times, color=colors)
    axes[2].set_xticks(x); axes[2].set_xticklabels(names, rotation=15)
    axes[2].set_ylabel('耗时 (ms)')
    axes[2].set_title('单次求解耗时')
    fig.suptitle('三方法对比', fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_divergence(err_div, path):
    """不收敛反例：A=[[1,2],[2,1]]（不严格对角占优）的 Jacobi 发散曲线。"""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(np.arange(1, len(err_div) + 1), err_div, '-o', color='#d62728',
                label='步差')
    ax.set_xlabel('迭代轮数')
    ax.set_ylabel('步差 ‖x_新 − x_旧‖∞')
    ax.set_title('不收敛反例：A=[[1,2],[2,1]]')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────
# GIF 动画（3 张独立 GIF：Jacobi / Gauss-Seidel / 直接法）
# 帧策略：把 n×n 内部解嵌入 (n+2)×(n+2) 含边界温度场，色标统一 RdBu_r 0–100。
# ────────────────────────────────────────────────────────────────────
def _embed_boundary(x_flat, n, T_top, T_others=0.0):
    """把 n×n 内部解嵌入 (n+2)×(n+2) 含边界温度场，便于画完整铁板。"""
    interior = x_flat.reshape(n, n)
    full = np.full((n+2, n+2), T_others, dtype=float)
    full[1:-1, 1:-1] = interior
    full[0, :] = T_top            # 顶边
    return full


def make_iter_gif(x_snaps, n, T_top, title, path, fps=8):
    """迭代过程 GIF：每帧一张嵌入后的温度场快照。"""
    frames = [_embed_boundary(s, n, T_top) for s in x_snaps]
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(frames[0], cmap='RdBu_r', vmin=0, vmax=100, origin='upper')
    ax.set_title(title); fig.colorbar(im, ax=ax, label='温度 °C')
    def update(k):
        im.set_data(frames[min(k, len(frames)-1)])
        ax.set_xlabel(f'第 {min(k,len(frames)-1)} 帧')
        return [im]
    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000//fps, blit=False)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def make_direct_gif(x_final, n, T_top, title, path, fps=2):
    """两帧：初值全 0 → 最终解。"""
    zero = np.zeros(n*n)
    make_iter_gif([zero, x_final], n, T_top, title, path, fps=fps)
