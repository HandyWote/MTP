"""任务三专用可视化风格：中文字体 + Okabe-Ito 色盲友好色板 + DPI=300。
本任务自包含，不与任务一/二共享。调用：from mission3 import viz_style; viz_style.apply()"""
import matplotlib as mpl

OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
             "#F0E442", "#56B4E9", "#E69F00", "#000000"]


def apply():
    mpl.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "figure.figsize": (7, 4.5),
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.prop_cycle": mpl.cycler(color=OKABE_ITO),
    })
