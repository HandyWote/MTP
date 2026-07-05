#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三控开关的设计与实现 -- 任务二
课程：数学思维实践（CST4822A）

逻辑：L = S1 xor S2 xor S3（三异或 = 按下奇数个开关则灯亮）
"任一开关变化 -> 输出变化"
"""

import itertools
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D

# ---------- 全局设置 ----------
# 中文字体（Windows 微软雅黑）
plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans SC", "DejaVu Sans"],
    "font.size": 13,
    "axes.unicode_minus": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
#  1. 核心逻辑
# ============================================================

def switch_logic(s1, s2, s3):
    """三控开关逻辑：L = S1 xor S2 xor S3"""
    return s1 ^ s2 ^ s3


def generate_truth_table():
    """返回 truth_table: list of (s1, s2, s3, L)"""
    return [(s1, s2, s3, switch_logic(s1, s2, s3))
            for s1, s2, s3 in itertools.product([0, 1], repeat=3)]


def print_truth_table():
    """打印格式对齐的真值表"""
    sep = "-" * 36
    print("\n" + sep)
    print("  三控开关真值表  (L = S1 xor S2 xor S3)")
    print(sep)
    print(f"{'S1':>4}  {'S2':>4}  {'S3':>4}  | {'L':>4}")
    print("-" * 24)
    for s1, s2, s3, L in generate_truth_table():
        print(f"{s1:>4}  {s2:>4}  {s3:>4}  | {L:>4}")
    print(sep)


# ============================================================
#  2. 交互模式
# ============================================================

def interactive_mode():
    """命令行交互：用户输入三个开关状态，输出灯的状态"""
    print("\n" + "=" * 50)
    print("  三控开关交互模式")
    print("  输入三个 0 或 1（空格分隔），如 1 0 1")
    print("  输入 q 退出")
    print("=" * 50)
    while True:
        raw = input("\n请输入 S1 S2 S3 > ").strip()
        if raw.lower() == "q":
            print("退出交互模式。")
            break
        parts = raw.split()
        if len(parts) != 3:
            print("[!] 请输入恰好三个值（用空格分隔）")
            continue
        try:
            s1, s2, s3 = [int(p) for p in parts]
        except ValueError:
            print("[!] 输入必须为整数 0 或 1")
            continue
        if not all(v in (0, 1) for v in (s1, s2, s3)):
            print("[!] 每个开关只能输入 0 或 1")
            continue
        L = switch_logic(s1, s2, s3)
        print(f"  S1={s1}, S2={s2}, S3={s3}  -->  灯 {'*亮' if L else 'o灭'}")


# ============================================================
#  3. 自动验证：任意开关翻转 -> 输出必翻转
# ============================================================

def auto_verify():
    """对 8 个初始状态 x 3 个开关各翻转一次，验证 L 是否翻转。返回验证结果列表"""
    truth_table = generate_truth_table()
    results = []  # 每一项: (s1,s2,s3, flipped_switch, L_before, L_after, passed)
    all_pass = True

    for s1, s2, s3, L in truth_table:
        state = [s1, s2, s3]
        for idx, switch_name in enumerate(["S1", "S2", "S3"]):
            before = L
            flipped = state.copy()
            flipped[idx] ^= 1  # 翻转
            after = switch_logic(*flipped)
            passed = (before != after)
            results.append((s1, s2, s3, switch_name, before, after, passed))
            if not passed:
                all_pass = False

    # 打印
    print("\n" + "=" * 60)
    print("  自动验证：翻转任一开关 -> 输出应翻转")
    print("=" * 60)
    passed_count = sum(1 for r in results if r[-1])
    for s1, s2, s3, sw, before, after, ok in results:
        mark = "[PASS]" if ok else "[FAIL]"
        print(f"  ({s1},{s2},{s3}) 翻转{sw}: L {before}->{after}  {mark}")

    if all_pass:
        print(f"\n  -- 结果: {passed_count}/{len(results)} 全部通过 [OK]")
    else:
        print(f"\n  -- 结果: {passed_count}/{len(results)} 存在失败 [FAIL]")
    return results


# ============================================================
#  4. 可视化
# ============================================================

# ---- 4a. 真值表颜色矩阵 (图 2-1) ----

def plot_truth_table_matrix():
    truth_table = generate_truth_table()  # list of (s1,s2,s3,L)
    data = np.array([[s1, s2, s3, L] for s1, s2, s3, L in truth_table])

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    cmap = matplotlib.colors.ListedColormap(["#E8F5E9", "#2E7D32"])  # 0=浅绿, 1=深绿
    im = ax.imshow(data.T, cmap=cmap, aspect="equal", vmin=0, vmax=1)

    # 格子内标注 0/1
    for i in range(8):            # 列 (状态)
        for j in range(4):        # 行 (变量)
            val = data[i, j]
            color = "white" if val == 1 else "#333"
            ax.text(i, j, str(val), ha="center", va="center", fontsize=12, fontweight="bold", color=color)

    ax.set_xticks(range(8))
    ax.set_xticklabels([f"({s1}{s2}{s3})" for s1, s2, s3, _ in truth_table], fontsize=9)
    ax.set_yticks(range(4))
    ax.set_yticklabels(["S1", "S2", "S3", "L"], fontsize=11)
    ax.set_title("图 2-1  三控开关真值表颜色矩阵", fontsize=14, fontweight="bold", pad=14)

    # 颜色条
    cbar = fig.colorbar(im, ax=ax, ticks=[0.25, 0.75], shrink=0.82, pad=0.02)
    cbar.ax.set_yticklabels(["0 (关/灭)", "1 (开/亮)"], fontsize=10)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "图2-1_真值表颜色矩阵.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] 已保存: {path}")


# ---- 4b. 状态空间图 -- 3D 立方体 (图 2-2) ----

def plot_state_cube():
    truth_table = generate_truth_table()

    fig = plt.figure(figsize=(6.5, 6))
    ax = fig.add_subplot(111, projection="3d")

    # 8 个顶点
    for s1, s2, s3, L in truth_table:
        color = "#FF6B6B" if L == 1 else "#4ECDC4"  # 红=亮, 青=灭
        ax.scatter(s1, s2, s3, c=color, s=180, edgecolors="black", linewidth=0.8, depthshade=True)

    # 标注每个顶点的 (S1,S2,S3) 和 L
    for s1, s2, s3, L in truth_table:
        label = f"({s1},{s2},{s3})\nL={L}"
        ax.text(s1, s2, s3 + 0.08, label, ha="center", va="bottom", fontsize=8, fontweight="bold")

    # 立方体的 12 条棱：哈密顿距离=1 的顶点对之间连线
    for (s1_a, s2_a, s3_a, _), (s1_b, s2_b, s3_b, _) in itertools.combinations(truth_table, 2):
        diff = abs(s1_a - s1_b) + abs(s2_a - s2_b) + abs(s3_a - s3_b)
        if diff == 1:
            ax.plot([s1_a, s1_b], [s2_a, s2_b], [s3_a, s3_b],
                    color="gray", linewidth=0.6, alpha=0.45)

    ax.set_xlabel("S1", fontsize=12, labelpad=8)
    ax.set_ylabel("S2", fontsize=12, labelpad=8)
    ax.set_zlabel("S3", fontsize=12, labelpad=8)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_zticks([0, 1])
    ax.set_title("图 2-2  三控开关状态空间图 (3D 立方体)", fontsize=14, fontweight="bold", pad=18)

    legend_elements = [
        Patch(facecolor="#FF6B6B", edgecolor="black", label="L=1 灯亮"),
        Patch(facecolor="#4ECDC4", edgecolor="black", label="L=0 灯灭"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=10)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "图2-2_状态空间图.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] 已保存: {path}")


# ---- 4c. 状态切换路径图 (图 2-3) ----

def plot_transition_paths():
    """在 3D 立方体上加边标注，标明每个边翻转了哪个开关"""
    truth_table = generate_truth_table()

    fig = plt.figure(figsize=(7, 6.5))
    ax = fig.add_subplot(111, projection="3d")

    # 顶点
    for s1, s2, s3, L in truth_table:
        color = "#FF6B6B" if L == 1 else "#4ECDC4"
        ax.scatter(s1, s2, s3, c=color, s=180, edgecolors="black", linewidth=0.8, depthshade=True)
        ax.text(s1, s2, s3 + 0.10, f"({s1},{s2},{s3})\nL={L}",
                ha="center", va="bottom", fontsize=8, fontweight="bold")

    # 棱 + 标注翻转哪个开关
    switch_names = ["S1", "S2", "S3"]
    for (a, _), (b, _) in itertools.combinations(
            [( (s1,s2,s3), L ) for s1,s2,s3,L in truth_table], 2):
        p1 = a
        p2 = b
        diff_vec = [abs(p1[i] - p2[i]) for i in range(3)]
        if sum(diff_vec) == 1:
            flipped_idx = diff_vec.index(1)
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                    color="#888888", linewidth=0.7, alpha=0.5, linestyle="--")
            # 在边的中点标注是哪个开关翻转
            mid = [(p1[i] + p2[i]) / 2 for i in range(3)]
            ax.text(mid[0], mid[1], mid[2], switch_names[flipped_idx],
                    ha="center", va="center", fontsize=9, fontweight="bold",
                    color="#1565C0",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85))

    ax.set_xlabel("S1", fontsize=12, labelpad=8)
    ax.set_ylabel("S2", fontsize=12, labelpad=8)
    ax.set_zlabel("S3", fontsize=12, labelpad=8)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_zticks([0, 1])
    ax.set_title("图 2-3  状态切换路径图 (边标注翻转的开关)", fontsize=13, fontweight="bold", pad=18)

    legend_elements = [
        Patch(facecolor="#FF6B6B", edgecolor="black", label="L=1 灯亮"),
        Patch(facecolor="#4ECDC4", edgecolor="black", label="L=0 灯灭"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=10)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "图2-3_状态切换路径图.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] 已保存: {path}")


# ---- 4d. 验证结果矩阵 (图 2-4) ----

def plot_verification_matrix(verify_results):
    """3x8 热力图：行 = 被翻转的开关 (S1/S2/S3)，列 = 8 种原始状态，颜色 = PASS/FAIL"""
    # 构建 3x8 矩阵：行 = 被翻转的开关, 列 = 8 种状态
    verify_matrix = np.zeros((3, 8), dtype=int)
    for s1, s2, s3, sw, _before, _after, passed in verify_results:
        col = s1 * 4 + s2 * 2 + s3 * 1  # 二进制编码到列索引
        row = {"S1": 0, "S2": 1, "S3": 2}[sw]
        verify_matrix[row, col] = 1 if passed else 0

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    cmap = matplotlib.colors.ListedColormap(["#EF5350", "#66BB6A"])  # 0=FAIL(红), 1=PASS(绿)
    im = ax.imshow(verify_matrix, cmap=cmap, aspect="equal", vmin=0, vmax=1)

    state_labels = ["000", "001", "010", "011", "100", "101", "110", "111"]
    for i in range(3):
        for j in range(8):
            status = "PASS" if verify_matrix[i, j] else "FAIL"
            color = "white"
            ax.text(j, i, status, ha="center", va="center", fontsize=9, fontweight="bold", color=color)

    ax.set_xticks(range(8))
    ax.set_xticklabels(state_labels, fontsize=10)
    ax.set_yticks(range(3))
    ax.set_yticklabels(["翻转 S1", "翻转 S2", "翻转 S3"], fontsize=11)
    ax.set_xlabel("原始状态 (S1 S2 S3)", fontsize=12, labelpad=8)
    ax.set_title("图 2-4  验证结果矩阵 (24 种翻转 -> 输出是否翻转)", fontsize=13, fontweight="bold", pad=14)

    cbar = fig.colorbar(im, ax=ax, ticks=[0.25, 0.75], shrink=0.85, pad=0.02)
    cbar.ax.set_yticklabels(["FAIL", "PASS"], fontsize=10)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "图2-4_验证结果矩阵.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] 已保存: {path}")


# ============================================================
#  5. 主入口
# ============================================================

def main():
    print("=" * 60)
    print("  任务二：三控开关的设计与实现")
    print("  逻辑: L = S1 xor S2 xor S3  (按下奇数个开关 -> 灯亮)")
    print("=" * 60)

    # --- 真值表 ---
    print_truth_table()

    # --- 自动验证 ---
    results = auto_verify()

    # --- 可视化 ---
    print("\n" + "=" * 40)
    print("  生成可视化图表 ...")
    print("=" * 40)
    plot_truth_table_matrix()
    plot_state_cube()
    plot_transition_paths()
    # 传入 auto_verify 结果避免重复验证
    plot_verification_matrix(results)

    print(f"\n{'='*40}")
    print(f"  全部完成！图表已保存至 {OUTPUT_DIR}/")
    print(f"{'='*40}")

    # 交互模式：如有需要，取消下面的注释
    # interactive_mode()


if __name__ == "__main__":
    main()
