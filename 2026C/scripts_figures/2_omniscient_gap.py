"""绘制问题二全知视角的对照图：累计购电费用与日末储电量。

数据来源：outputs/tables/2_全知视角每日对比.csv（由 scripts_tables/2_omniscient.py 生成）。
左图用累计曲线显示 L0 因果滚动、L1 完美日预测同策略与 L2 全知全局下界的费用差距；
右图用日末储电量显示三种口径的跨日调度差异。同时输出 300 dpi PNG 与 SVG。
"""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MultipleLocator


PROJECT_DIR = Path(__file__).resolve().parents[1]
DAILY_FILE = PROJECT_DIR / "outputs" / "tables" / "2_全知视角每日对比.csv"
FIGURE_DIR = PROJECT_DIR / "outputs" / "figures"

COLOR_L0 = "#C0392B"
COLOR_L1 = "#2667A6"
COLOR_L2 = "#D97924"
COLOR_BAND = "#F2D9C9"


def configure_style() -> None:
    """自动选择已安装的中文字体，避免中文显示为方框。"""
    candidates = (
        "Noto Sans CJK SC", "Source Han Sans SC", "Microsoft YaHei",
        "SimHei", "PingFang SC", "Heiti SC", "Arial Unicode MS",
    )
    installed = {font.name for font in font_manager.fontManager.ttflist}
    chinese_font = next((name for name in candidates if name in installed), None)
    if chinese_font is None:
        raise RuntimeError("未找到中文字体，请安装 Noto Sans CJK SC、微软雅黑或黑体后重试。")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [chinese_font, "DejaVu Sans"],
        "axes.unicode_minus": False,
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "axes.labelpad": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 10,
        "legend.fontsize": 10.5,
        "lines.linewidth": 2.0,
        "savefig.dpi": 300,
        "svg.fonttype": "path",  # SVG 中保留字形，跨设备查看无需另装字体。
    })


def load_daily() -> tuple[list[date], list[float], list[float], list[float],
                          list[float], list[float], list[float]]:
    """读取逐日对比表，返回日期与六个序列。"""
    if not DAILY_FILE.is_file():
        raise RuntimeError(
            f"未找到 {DAILY_FILE}。请先运行 scripts_tables/2_omniscient.py。"
        )
    days: list[date] = []
    cumulative_l0: list[float] = []
    cumulative_l1: list[float] = []
    cumulative_l2: list[float] = []
    end_l0: list[float] = []
    end_l1: list[float] = []
    end_l2: list[float] = []
    with DAILY_FILE.open(encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            days.append(date.fromisoformat(row["日期"]))
            cumulative_l0.append(float(row["L0 累计购电费（元）"]))
            cumulative_l1.append(float(row["L1 累计购电费（元）"]))
            cumulative_l2.append(float(row["L2 累计购电费（元）"]))
            end_l0.append(float(row["L0 实际日末储电量（kWh）"]))
            end_l1.append(float(row["L1 日末储电量（kWh）"]))
            end_l2.append(float(row["L2 日末储电量（kWh）"]))
    if not days:
        raise RuntimeError(f"{DAILY_FILE.name} 没有数据行。")
    return days, cumulative_l0, cumulative_l1, cumulative_l2, end_l0, end_l1, end_l2


def month_ticks(days: list[date]) -> tuple[list[int], list[str]]:
    """给出每个自然月首日的横坐标与标签。"""
    positions: list[int] = []
    labels: list[str] = []
    for index, day in enumerate(days):
        if day.day == 1:
            positions.append(index)
            labels.append(f"{day.month}月")
    positions.append(len(days) - 1)
    labels.append("12/31")
    return positions, labels


def create_axes(title: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    fig.subplots_adjust(left=0.085, right=0.965, bottom=0.16, top=0.88)
    ax.set_title(title, loc="left", pad=14, fontweight="semibold")
    ax.set_ylabel(ylabel)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#DEE4EB", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#9AA6B2")
    ax.tick_params(axis="both", which="both", color="#9AA6B2")
    return fig, ax


def save_figure(fig, name: str) -> None:
    for extension in ("png", "svg"):
        output_file = FIGURE_DIR / f"{name}.{extension}"
        fig.savefig(output_file, facecolor="white")
        print(f"已保存：{output_file}")
    plt.close(fig)


def main() -> None:
    days, cumulative_l0, cumulative_l1, cumulative_l2, end_l0, end_l1, end_l2 = load_daily()
    configure_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    positions, labels = month_ticks(days)
    x = list(range(len(days)))

    wan_l0 = [value / 1e4 for value in cumulative_l0]
    wan_l1 = [value / 1e4 for value in cumulative_l1]
    wan_l2 = [value / 1e4 for value in cumulative_l2]

    fig, ax = create_axes(
        "问题二全知视角：累计购电费用（2025-02-01 至 12-31）", "累计购电费（万元）"
    )
    ax.fill_between(x, wan_l2, wan_l0, color=COLOR_BAND, alpha=0.55, linewidth=0,
                    label="正式方案相对理论下界的差距")
    ax.plot(x, wan_l0, color=COLOR_L0, label="L0 因果滚动实际执行（正式方案）")
    ax.plot(x, wan_l1, color=COLOR_L1, linestyle="--", label="L1 全知日预测同策略")
    ax.plot(x, wan_l2, color=COLOR_L2, linestyle="-.", label="L2 全知全局理论下界")
    gap = wan_l0[-1] - wan_l2[-1]
    ax.annotate(
        f"期末差距 {gap:.1f} 万元\n（占正式方案 {gap / wan_l0[-1] * 100:.2f}%）",
        xy=(x[-1], (wan_l0[-1] + wan_l2[-1]) / 2),
        xytext=(-118, -66), textcoords="offset points",
        fontsize=10.5, color="#7A4B2A",
        arrowprops=dict(arrowstyle="-", color="#B08A6A", linewidth=1.0),
    )
    ax.set_xticks(positions, labels)
    ax.set_xlim(0, x[-1])
    ax.set_ylim(min(wan_l2) - 5, max(wan_l0) + 12)
    ax.legend(loc="upper left", frameon=False)
    save_figure(fig, "2_全知视角累计购电费用对比")

    fig, ax = create_axes(
        "问题二全知视角：日末储电量对比（2025-02-01 至 12-31）", "日末储电量（kWh）"
    )
    ax.axhspan(0, 1200, color="#E8EDF2", alpha=0.7, linewidth=0)
    ax.axhspan(10800, 12000, color="#E8EDF2", alpha=0.7, linewidth=0)
    ax.axhline(6000, color="#98A6B3", linestyle=":", linewidth=1.2)
    ax.plot(x, end_l0, color=COLOR_L0, label="L0 因果滚动实际执行（正式方案）")
    ax.plot(x, end_l1, color=COLOR_L1, linestyle="--", label="L1 全知日预测同策略")
    ax.plot(x, end_l2, color=COLOR_L2, linestyle="-.", label="L2 全知全局理论下界")
    ax.set_xticks(positions, labels)
    ax.set_xlim(0, x[-1])
    ax.set_ylim(0, 12000)
    ax.yaxis.set_major_locator(MultipleLocator(2000))
    ax.text(x[-1] - 60, 5820, "日末参考值 6000 kWh", fontsize=9.5, color="#5D6975", ha="right", va="top")
    ax.legend(loc="upper left", frameon=False, ncol=1)
    fig.text(
        0.085, 0.035,
        "灰色区间为储能设备安全运行范围外的 0-1200 kWh 与 10800-12000 kWh；"
        "数据来源：outputs/tables/2_全知视角每日对比.csv。",
        fontsize=8.5, color="#5D6975",
    )
    save_figure(fig, "2_全知视角日末储电量对比")

    print(f"绘图完成：{len(days)} 天，2 张图，各保存 PNG 和 SVG。")


if __name__ == "__main__":
    main()
