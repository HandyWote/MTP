"""一键复现：跑测试 → 跑实验 → 出图。
可从仓库根运行：python -m mission3.run_all
也可从 mission3/ 目录运行：python run_all.py  或  python -m run_all"""
import subprocess
import sys
from pathlib import Path

# 让 mission3 包无论从哪个 cwd 运行都可导入（把仓库根加入 sys.path）
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_TESTS = str(Path(__file__).resolve().parent / "tests")


def main():
    subprocess.run([sys.executable, "-m", "pytest", _TESTS, "-v"], check=False)
    from mission3.experiments import run_all_experiments
    run_all_experiments()
    from mission3 import plot_matplotlib
    plot_matplotlib.main()
    print("\nDone. 见 mission3/output/ 与 experiments.json")


if __name__ == "__main__":
    main()
