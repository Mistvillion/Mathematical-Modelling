"""问题四（问题二部分）补充分析：价格预测口径与分位数敏感性的因果对比。

本脚本不改变问题四（问题二部分）的正式答案，只把同一评价区间、同一实际结算电价、
同一负载/光伏预测与执行规则下的不同价格预测口径放在一起比较。正式结果仍以
scripts_tables/4-2_price_rolling_optimization.py 的输出为准。

比较口径（每个口径都会完整重跑一遍 365 天滚动过程）：

1. 本方案        ：水平 × 形态价格预测（文档第 6.1.4 节），主口径负载 0.8、光伏 0.2；
2. 参考曲线口径  ：把每天的价格预测固定为附件 1 的参考曲线，完全不使用历史电价；
3. 前一日曲线口径：把每天的价格预测固定为前一日的实际电价曲线（首日为参考曲线）；
4. 完美价格预见  ：计划与日内目标直接使用当天实际电价（事后参考，不可实施）；
5. 分位数敏感性  ：价格预测同本方案，但把负载分位数改为 0.82、光伏分位数改为 0.18，
                  用于检验第 6.3 节价格–净需求正相关修正的因果效果。

运行（需先完成问题四（问题二部分）主脚本，保证附件与输出目录存在）：
    python scripts_tables/4-2_price_forecast_baselines.py

输出：
    outputs/tables/4-2_价格预测口径对比.csv

单次滚动约 1 分钟，全部五个口径合计约 5 分钟。
"""

from __future__ import annotations

import csv
import importlib.util
import sys
import time
from pathlib import Path
from types import ModuleType

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
MAIN_SCRIPT = PROJECT_DIR / "scripts_tables" / "4-2_price_rolling_optimization.py"
OUTPUT_FILE = PROJECT_DIR / "outputs" / "tables" / "4-2_价格预测口径对比.csv"

MODES = (
    ("本方案（水平 × 形态）", "main", (0.80, 0.20)),
    ("参考曲线口径", "reference", (0.80, 0.20)),
    ("前一日曲线口径", "yesterday", (0.80, 0.20)),
    ("完美价格预见（事后参考）", "perfect", (0.80, 0.20)),
    ("分位数敏感性 0.82/0.18", "main", (0.82, 0.18)),
)


def load_main_module() -> ModuleType:
    """按文件路径加载 4-2 主脚本，避免脚本名中的连字符影响 import。"""
    spec = importlib.util.spec_from_file_location("price_rolling", MAIN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载主脚本：{MAIN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["price_rolling"] = module
    spec.loader.exec_module(module)
    return module


def run_mode(module: ModuleType, data: object, mode: str, quantiles: tuple[float, float]):
    """在指定价格口径与分位数下完整重跑全年滚动过程。"""
    load_quantile, photovoltaic_quantile = quantiles
    original_price_candidates = module.price_candidates
    original_load_quantile = module.LOAD_QUANTILE
    original_photovoltaic_quantile = module.PHOTOVOLTAIC_QUANTILE
    day_index = {day: index for index, day in enumerate(data.dates)}

    def patched_price_candidates(target, history_dates, historical_price, reference_price):
        index = day_index[target]
        if mode == "main":
            return original_price_candidates(
                target, history_dates, historical_price, reference_price
            )
        if mode == "reference":
            forecast = data.reference_price.copy()
        elif mode == "yesterday":
            forecast = (
                data.reference_price.copy()
                if index == 0
                else data.actual_price[index - 1].copy()
            )
        elif mode == "perfect":
            forecast = data.actual_price[index].copy()
        else:
            raise ValueError(f"未知价格口径：{mode}")
        return np.tile(forecast, (module.PRICE_CANDIDATE_COUNT, 1))

    module.price_candidates = patched_price_candidates
    module.LOAD_QUANTILE = load_quantile
    module.PHOTOVOLTAIC_QUANTILE = photovoltaic_quantile
    try:
        rolling = module.solve_rolling_model(data, module.ForecastEngine())
        module.validate_rolling_result(data, rolling)
    finally:
        module.price_candidates = original_price_candidates
        module.LOAD_QUANTILE = original_load_quantile
        module.PHOTOVOLTAIC_QUANTILE = original_photovoltaic_quantile
    return rolling


def summarize(module: ModuleType, rolling: object) -> dict[str, float]:
    """汇总正式期的费用、紧急购电、弃光、终端状态与价格预测误差。"""
    days = rolling.official_days
    forecast_price = np.concatenate([day.forecast.price for day in days])
    actual_price = np.concatenate([day.actual_price for day in days])
    planned_cost = sum(day.planned_purchase_cost for day in days)
    forecast_planned_cost = sum(day.forecast_planned_purchase_cost for day in days)
    emergency_cost = sum(day.emergency_purchase_cost for day in days)
    return {
        "price_mae": float(np.mean(np.abs(actual_price - forecast_price))),
        "price_rmse": float(np.sqrt(np.mean((actual_price - forecast_price) ** 2))),
        "planned_cost": planned_cost,
        "forecast_planned_cost": forecast_planned_cost,
        "emergency_cost": emergency_cost,
        "total_cost": planned_cost + emergency_cost,
        "emergency_energy": sum(float(day.replay.emergency_purchase.sum()) for day in days),
        "emergency_periods": int(sum(
            int((day.replay.emergency_purchase > module.EMERGENCY_TOLERANCE).sum())
            for day in days
        )),
        "emergency_intervals": sum(
            len(module.group_emergency_intervals(day.replay.emergency_purchase))
            for day in days
        ),
        "curtailment": sum(float(day.replay.curtailment.sum()) for day in days),
        "unused_planned": sum(
            float(day.replay.unused_planned_purchase.sum()) for day in days
        ),
        "planned_end": float(days[-1].plan.stored_energy[-1]),
        "actual_end": float(days[-1].replay.stored_energy[-1]),
        "terminal_error": module.terminal_interval_error(days[-1].replay.stored_energy[-1]),
        "warmup_cost": sum(day.realized_total_cost for day in rolling.warmup_days),
    }


def main() -> None:
    """逐个口径重跑滚动过程，写出对比 CSV 并打印摘要。"""
    if Path(sys.prefix).name != "2026C" or not (Path(sys.prefix) / "conda-meta").is_dir():
        raise RuntimeError("项目要求使用 conda 环境 2026C。")
    module = load_main_module()
    module.validate_terminal_bounds()
    data = module.load_model_data()

    header = (
        "口径",
        "负载分位数",
        "光伏分位数",
        "电价MAE（元/kWh）",
        "电价RMSE（元/kWh）",
        "计划购电费预测口径（元）",
        "计划购电费实际口径（元）",
        "紧急购电费（元）",
        "总购电费（元）",
        "紧急购电量（kWh）",
        "紧急购电时段数",
        "连续紧急购电区间数",
        "弃光总量（kWh）",
        "未利用计划购电量（kWh）",
        "年末计划储电量（kWh）",
        "年末实际储电量（kWh）",
        "年末实际储电量越界量（kWh）",
        "1月初始化费用（元）",
    )
    rows: list[tuple[object, ...]] = []
    for label, mode, quantiles in MODES:
        started = time.time()
        rolling = run_mode(module, data, mode, quantiles)
        summary = summarize(module, rolling)
        rows.append((
            label,
            f"{quantiles[0]:.2f}",
            f"{quantiles[1]:.2f}",
            f"{summary['price_mae']:.6f}",
            f"{summary['price_rmse']:.6f}",
            f"{summary['forecast_planned_cost']:.6f}",
            f"{summary['planned_cost']:.6f}",
            f"{summary['emergency_cost']:.6f}",
            f"{summary['total_cost']:.6f}",
            f"{summary['emergency_energy']:.6f}",
            str(summary["emergency_periods"]),
            str(summary["emergency_intervals"]),
            f"{summary['curtailment']:.6f}",
            f"{summary['unused_planned']:.6f}",
            f"{summary['planned_end']:.6f}",
            f"{summary['actual_end']:.6f}",
            f"{summary['terminal_error']:.6f}",
            f"{summary['warmup_cost']:.6f}",
        ))
        print(
            f"{label}：总购电费 {summary['total_cost']:.2f} 元，"
            f"电价 MAE {summary['price_mae']:.6f} 元/kWh，"
            f"用时 {time.time() - started:.0f}s",
            flush=True,
        )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"已保存：{OUTPUT_FILE.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
