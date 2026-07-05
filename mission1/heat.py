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
    # symlog：让 Jacobi/GS(≈1e-6) 与直接法(0) 的柱子都可见
    axes[1].set_yscale('symlog', linthresh=1e-9)
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
    if not x_snaps:
        raise ValueError("x_snaps 不能为空（至少需含初值副本）")
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


# ────────────────────────────────────────────────────────────────────
# main：端到端串联
# 流水线：建模（1D n=5 + 2D 2×2 + 2D 30×30）→ 三方法求解（含计时）
#         → 6 张静态图 + 3 张 GIF → 终端打印报告
# ────────────────────────────────────────────────────────────────────
def main():
    import os
    import time

    out_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(out_dir, exist_ok=True)

    print('=' * 64)
    print('任务一：迭代法解稳态热传导 Ax=b')
    print('=' * 64)

    # ── 1. 建模 ────────────────────────────────────────────────────
    print('\n[1/5] 建模')
    # 铁棍 n=5 内部点，两端 100°C / 0°C
    A1, b1 = build_1d(n=5, T_left=100, T_right=0)
    # 铁板 2×2，上边 100°C，其余 0°C（最小可见样例）
    A2_2, b2_2 = build_2d(n=2, T_top=100, T_others=0)
    # 铁板 30×30（主算例，用于误差曲线 / 谱半径 / GIF）
    N = 30
    A, b = build_2d(n=N, T_top=100, T_others=0)
    print(f'  铁棍 1D n=5        : A{A1.shape}')
    print(f'  铁板 2D 2×2        : A{A2_2.shape}')
    print(f'  铁板 2D {N}×{N}        : A{A.shape}')

    # ── 2. 三方法求解（计时）──────────────────────────────────────
    print('\n[2/5] 求解（Jacobi / Gauss-Seidel / 直接法）')
    x0_big = np.zeros(N * N)

    t0 = time.perf_counter()
    x_j, err_j, snaps_j = jacobi(A, b, x0_big, snapshot_every=50)
    t_j = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    x_g, err_g, snaps_g = gauss_seidel(A, b, x0_big, snapshot_every=50)
    t_g = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    x_d = solve_direct(A, b)
    t_d = (time.perf_counter() - t0) * 1000

    # 真误差：迭代解 vs 直接解（裁判）最大偏差
    true_err_j = float(np.max(np.abs(x_j - x_d)))
    true_err_g = float(np.max(np.abs(x_g - x_d)))

    stats = {
        'Jacobi':       (len(err_j), float(err_j[-1]), t_j),
        'Gauss-Seidel': (len(err_g), float(err_g[-1]), t_g),
        '直接法':        (0, 0.0, t_d),
    }
    print(f"  Jacobi      : {len(err_j):4d} 轮, 步差={err_j[-1]:.2e}, "
          f"真误差={true_err_j:.2e}, 耗时={t_j:.1f} ms")
    print(f"  Gauss-Seidel: {len(err_g):4d} 轮, 步差={err_g[-1]:.2e}, "
          f"真误差={true_err_g:.2e}, 耗时={t_g:.1f} ms")
    print(f"  直接法       :    0 轮, 真误差=0（裁判基准）, 耗时={t_d:.1f} ms")

    # ── 3. 6 张静态图 ──────────────────────────────────────────────
    print('\n[3/5] 生成 6 张静态图')
    # 直接解作为裁判
    x_rod = solve_direct(A1, b1)
    x_2x2 = solve_direct(A2_2, b2_2)

    plot_rod(x_rod, os.path.join(out_dir, 'fig1_rod.png'))
    plot_matrix_and_bars(A2_2, x_2x2, os.path.join(out_dir, 'fig2_matrix.png'))
    plot_heatmap_compare(x_j, x_d, N, os.path.join(out_dir, 'fig3_heatmap.png'))
    plot_error_curve(err_j, err_g, os.path.join(out_dir, 'fig4_error.png'))
    plot_method_bars(stats, os.path.join(out_dir, 'fig5_bars.png'))

    # 发散反例（brief 笔误修正：jacobi 第 3 个位置参数 x0 不能漏）
    A_div = np.array([[1, 2], [2, 1]], dtype=float)
    _, err_div, _ = jacobi(A_div, np.array([1., 1.]), np.zeros(2), max_iter=30)
    plot_divergence(err_div, os.path.join(out_dir, 'fig6_diverge.png'))
    print('  ✓ fig1–fig6 已写入 output/')

    # ── 4. 3 张 GIF ────────────────────────────────────────────────
    print('\n[4/5] 生成 3 张 GIF')
    make_iter_gif(snaps_j, N, 100, 'Jacobi 迭代过程',
                  os.path.join(out_dir, 'jacobi.gif'), fps=8)
    make_iter_gif(snaps_g, N, 100, 'Gauss-Seidel 迭代过程',
                  os.path.join(out_dir, 'gauss_seidel.gif'), fps=8)
    make_direct_gif(x_d, N, 100, '直接解（一帧到位）',
                    os.path.join(out_dir, 'direct.gif'), fps=2)
    print('  ✓ jacobi.gif / gauss_seidel.gif / direct.gif 已写入 output/')

    # ── 5. 终端报告 ────────────────────────────────────────────────
    print('\n[5/5] 报告')
    print('-' * 64)
    print('收敛轮数')
    print(f'  Jacobi       : {len(err_j)} 轮')
    print(f'  Gauss-Seidel : {len(err_g)} 轮  （约为 Jacobi 的 '
          f'{len(err_g)/len(err_j):.1%}）')
    print('\n谱半径（迭代矩阵收敛速率，越小越快；< 1 才收敛）')
    rho_j_theory = float(np.cos(np.pi / (N + 1)))         # cos(π/31)
    rho_g_theory = rho_j_theory ** 2                       # ρ_J²
    rho_j_call = spectral_radius_jacobi(N)
    rho_g_call = spectral_radius_gs(N)
    print(f'  ρ_J  理论 cos(π/31)     = {rho_j_theory:.6f}')
    print(f'       spectral_radius_* = {rho_j_call:.6f}')
    print(f'  ρ_GS 理论 cos²(π/31)   = {rho_g_theory:.6f}')
    print(f'       spectral_radius_* = {rho_g_call:.6f}')
    print('\n真误差（迭代解 vs 直接解，max 偏差）')
    print(f'  Jacobi       : {true_err_j:.2e}  '
          f'(理论 tol/(1−ρ_J) ≈ {1e-6/(1-rho_j_theory):.2e})')
    print(f'  Gauss-Seidel : {true_err_g:.2e}  '
          f'(理论 tol/(1−ρ_GS) ≈ {1e-6/(1-rho_g_theory):.2e})')
    print('\n输出文件：6 PNG + 3 GIF，共 9 个，位于 output/')
    print('=' * 64)


if __name__ == '__main__':
    main()
