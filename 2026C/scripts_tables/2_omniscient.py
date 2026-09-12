"""问题二补充分析：全知视角下的全局最优下界与预测信息价值分解。

本脚本不改变问题二的正式答案，只做事后对照，输出写入 outputs/tables：

1. L2 全知全局联合优化（理论下界）
   把评价期内每个 10 分钟时段的实际负载、实际光伏（附件 2）与已知电价一次性交给
   同一个模型，在整个评价时段上联合优化购电与储能，得到完美信息下确定性问题的
   全局最优。由于它忽略了“计划日当天及以后数据不可用”的非前视约束，任何满足同一
   物理约束与同一年末区间、且只按实际购电与紧急购电付费的因果方案，其实际费用都
   不低于该值，因此它是问题二正式方案的理论下界（不可实施，只用于衡量差距）。

2. L1 全知日预测同策略
   只把逐日预测替换为当日实测值，其余策略结构（0:00 冻结全天计划购电量、日内滚动
   执行储能、日末储电量软目标、年末储电量区间）与正式方案完全一致。该口径用于把
   正式方案与理论下界的差距分解为“预测误差代价”和“逐日结构代价”两部分。

3. L0 因果滚动结果
   直接读取 outputs/tables/2_每日运行审计.csv 与 2_滚动预测与调度明细.csv，作为
   对照口径，不重新计算；问题二的正式答案仍以 scripts_tables/2_rolling_optimization.py
   的输出为准。

运行（需先完成问题二主脚本，保证审计文件存在）：
    python scripts_tables/2_omniscient.py

输出：
    outputs/tables/2_全知视角逐时段明细.csv
    outputs/tables/2_全知视角每日对比.csv
    outputs/tables/2_全知视角口径对比.csv
    outputs/tables/2_全知视角信息价值分解.csv
    outputs/tables/2_全知视角校验.csv
"""

from __future__ import annotations

import csv
import importlib.util
import sys
import time
from dataclasses import dataclass
from datetime import date
from math import isfinite
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, vstack as sparse_vstack


PROJECT_DIR = Path(__file__).resolve().parents[1]
ROLLING_SCRIPT = PROJECT_DIR / "scripts_tables" / "2_rolling_optimization.py"
OUTPUT_TABLE_DIR = PROJECT_DIR / "outputs" / "tables"
AUDIT_FILE = OUTPUT_TABLE_DIR / "2_每日运行审计.csv"
DETAIL_FILE = OUTPUT_TABLE_DIR / "2_滚动预测与调度明细.csv"
CHECK_FILE = OUTPUT_TABLE_DIR / "2_模型校验.csv"

OMNI_DETAIL_FILE = OUTPUT_TABLE_DIR / "2_全知视角逐时段明细.csv"
OMNI_DAILY_FILE = OUTPUT_TABLE_DIR / "2_全知视角每日对比.csv"
OMNI_CALIBER_FILE = OUTPUT_TABLE_DIR / "2_全知视角口径对比.csv"
OMNI_DECOMPOSITION_FILE = OUTPUT_TABLE_DIR / "2_全知视角信息价值分解.csv"
OMNI_CHECK_FILE = OUTPUT_TABLE_DIR / "2_全知视角校验.csv"

# 第二阶段（最小化充放电总吞吐）允许的费用上浮；由小到大取第一个可行值，
# 只用于吸收求解器数值误差，不影响第一阶段给出的理论下界。
COST_TOLERANCE_CANDIDATES = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
# 全知模型自身约束的可行性容差，量级与主脚本一致。
OMNI_TOLERANCE = 1e-5

# 由 load_rolling_module 赋值，供本模块的 CSV 写出函数复用主脚本的格式约定。
ROLLING: ModuleType | None = None


@dataclass(frozen=True)
class CausalDay:
    """L0 因果方案某一天的实际运行结果（来自每日运行审计 CSV）。"""

    day: date
    initial_energy: float
    end_energy: float
    planned_cost: float
    emergency_cost: float
    total_cost: float
    emergency_energy: float


@dataclass(frozen=True)
class PerfectForecastDay:
    """L1 完美日预测同策略下某一天的计划与执行结果。"""

    day: date
    initial_energy: float
    end_energy: float
    planned_cost: float
    emergency_cost: float
    total_cost: float
    grid_purchase: np.ndarray
    charge: np.ndarray
    discharge: np.ndarray
    surplus: np.ndarray
    curtailment: np.ndarray
    unused_purchase: np.ndarray
    emergency_purchase: np.ndarray


@dataclass(frozen=True)
class OmniscientSolution:
    """L2 全知全局联合优化问题的解与诊断。"""

    scope_label: str
    day_count: int
    period_count: int
    initial_energy: float
    terminal_interval: tuple[float, float]
    lower_bound: float
    achieved_cost: float
    cost_tolerance: float
    throughput: float
    grid_purchase: np.ndarray
    charge: np.ndarray
    discharge: np.ndarray
    surplus: np.ndarray
    curtailment: np.ndarray
    unused_purchase: np.ndarray
    stored_energy: np.ndarray
    max_balance_residual: float
    max_state_residual: float
    max_surplus_violation: float
    max_unused_violation: float
    max_simultaneous: float
    min_stored_energy: float
    max_stored_energy: float
    terminal_energy: float
    solve_seconds: float


def check_environment() -> None:
    """与主脚本一致：要求使用项目约定的 conda 环境 2026C。"""
    if Path(sys.prefix).name != "2026C" or not (Path(sys.prefix) / "conda-meta").is_dir():
        raise RuntimeError("项目要求使用 conda 环境 2026C。")


def load_rolling_module() -> ModuleType:
    """按文件路径导入问题二主脚本，只复用其函数与常量，不触发其 main。"""
    global ROLLING
    if not ROLLING_SCRIPT.is_file():
        raise RuntimeError(f"未找到问题二主脚本：{ROLLING_SCRIPT}")
    spec = importlib.util.spec_from_file_location("q2_rolling_optimization", ROLLING_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法导入问题二主脚本：{ROLLING_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    ROLLING = module
    return module


def require_rolling() -> ModuleType:
    """返回已导入的问题二主脚本模块。"""
    if ROLLING is None:
        raise RuntimeError("问题二主脚本尚未导入，请先调用 load_rolling_module。")
    return ROLLING


def write_csv(
    path: Path,
    header: Sequence[str],
    rows: Iterable[Sequence[object]],
) -> Path:
    """沿用主脚本的 CSV 写出约定（UTF-8 BOM、LF 换行）。"""
    return require_rolling().write_csv(path, header, rows)


def display_number(value: float) -> str:
    """沿用主脚本的展示精度（保留 6 位小数并清除负零）。"""
    return require_rolling().display_number(value)


def split_surplus(
    surplus: np.ndarray,
    photovoltaic: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """按“先弃光”的统一口径把剩余电量拆成弃光量和未利用购电量。"""
    curtailment = np.minimum(surplus, photovoltaic)
    unused_purchase = surplus - curtailment
    return curtailment, unused_purchase


def read_causal_audit(january_days: int) -> tuple[dict[date, CausalDay], list[CausalDay]]:
    """读取问题二每日运行审计，返回全日期索引与正式期逐日列表。"""
    if not AUDIT_FILE.is_file():
        raise RuntimeError(
            f"未找到 {AUDIT_FILE}。请先运行 scripts_tables/2_rolling_optimization.py。"
        )
    required = (
        "日期", "日初储电量", "实际日末储电量", "计划购电费", "紧急购电费",
        "实际总购电费", "紧急购电量",
    )
    days: list[CausalDay] = []
    with AUDIT_FILE.open(encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None or any(name not in reader.fieldnames for name in required):
            raise RuntimeError(f"{AUDIT_FILE.name} 缺少必要列：{required}")
        for row in reader:
            days.append(
                CausalDay(
                    day=date.fromisoformat(row["日期"]),
                    initial_energy=float(row["日初储电量"]),
                    end_energy=float(row["实际日末储电量"]),
                    planned_cost=float(row["计划购电费"]),
                    emergency_cost=float(row["紧急购电费"]),
                    total_cost=float(row["实际总购电费"]),
                    emergency_energy=float(row["紧急购电量"]),
                )
            )
    if len(days) <= january_days:
        raise RuntimeError(f"{AUDIT_FILE.name} 的日期数量不足。")
    if len({item.day for item in days}) != len(days):
        raise RuntimeError(f"{AUDIT_FILE.name} 存在重复日期。")
    official = days[january_days:]
    if len(official) != require_rolling().OFFICIAL_DAYS:
        raise RuntimeError(f"{AUDIT_FILE.name} 的正式期天数不是 {require_rolling().OFFICIAL_DAYS}。")
    return {item.day: item for item in days}, official


def read_causal_detail() -> tuple[dict[str, float], float]:
    """汇总因果方案逐时段明细的电量口径与逐时段费用之和，用于交叉校验。"""
    wanted = {
        "planned_purchase": "计划购电量（kWh）",
        "charge": "实际充电量（kWh）",
        "discharge": "实际放电量（kWh）",
        "emergency": "紧急购电量（kWh）",
        "curtailment": "实际弃光量（kWh）",
        "unused": "未利用计划购电量（kWh）",
    }
    totals = dict.fromkeys(wanted, 0.0)
    detail_cost = 0.0
    if not DETAIL_FILE.is_file():
        raise RuntimeError(
            f"未找到 {DETAIL_FILE}。请先运行 scripts_tables/2_rolling_optimization.py。"
        )
    with DETAIL_FILE.open(encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required = (*wanted.values(), "时段总购电费（元）")
        if reader.fieldnames is None or any(name not in reader.fieldnames for name in required):
            raise RuntimeError(f"{DETAIL_FILE.name} 缺少必要列。")
        for row in reader:
            for key, column in wanted.items():
                totals[key] += float(row[column])
            detail_cost += float(row["时段总购电费（元）"])
    return totals, detail_cost


def read_causal_check_totals() -> dict[str, float]:
    """读取问题二模型校验 CSV 的权威总量（未经展示舍入累积）。"""
    mapping = {
        "planned_purchase": "计划购电总量",
        "emergency": "紧急购电总量",
        "curtailment": "弃光总量",
        "unused": "未利用计划购电总量",
        "total_cost": "334天总费用",
    }
    if not CHECK_FILE.is_file():
        raise RuntimeError(
            f"未找到 {CHECK_FILE}。请先运行 scripts_tables/2_rolling_optimization.py。"
        )
    wanted = set(mapping.values())
    found: dict[str, float] = {}
    with CHECK_FILE.open(encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None or any(
            name not in reader.fieldnames for name in ("校验项", "数值")
        ):
            raise RuntimeError(f"{CHECK_FILE.name} 缺少必要列。")
        for row in reader:
            if row["校验项"] in wanted:
                found[row["校验项"]] = float(row["数值"])
    missing = wanted.difference(found)
    if missing:
        raise RuntimeError(f"{CHECK_FILE.name} 缺少校验项：" + "、".join(sorted(missing)))
    return {key: found[name] for key, name in mapping.items()}


def validate_horizon_inputs(
    load_energy: np.ndarray,
    photovoltaic_energy: np.ndarray,
    price: np.ndarray,
    initial_energy: float,
    terminal_interval: tuple[float, float],
    rolling: ModuleType,
) -> int:
    """校验全时段联合优化的输入，返回时段数。"""
    if not (
        load_energy.shape == photovoltaic_energy.shape == price.shape
        and load_energy.ndim == 1
        and load_energy.size > 0
        and load_energy.size % rolling.N_PERIODS == 0
    ):
        raise ValueError("全知模型的负载、光伏与电价必须是等长的整日序列。")
    for name, values in (
        ("负载", load_energy), ("光伏", photovoltaic_energy), ("电价", price)
    ):
        if not np.isfinite(values).all():
            raise ValueError(f"全知模型的{name}包含非有限值。")
    if np.any(load_energy < 0) or np.any(photovoltaic_energy < 0):
        raise ValueError("全知模型的负载与光伏必须非负。")
    if np.any(price <= 0):
        raise ValueError("全知模型要求电价严格为正。")
    if not rolling.E_MIN - rolling.BOUND_TOLERANCE <= initial_energy <= rolling.E_MAX + rolling.BOUND_TOLERANCE:
        raise ValueError("全知模型的初始储电量超出设备安全范围。")
    lower, upper = terminal_interval
    if not (
        rolling.E_MIN - rolling.BOUND_TOLERANCE
        <= lower
        <= upper
        <= rolling.E_MAX + rolling.BOUND_TOLERANCE
    ):
        raise ValueError("全知模型的年末区间必须包含在设备安全范围内且上下限有序。")
    return int(load_energy.size)


def solve_omniscient_horizon(
    scope_label: str,
    load_energy: np.ndarray,
    photovoltaic_energy: np.ndarray,
    price: np.ndarray,
    initial_energy: float,
    terminal_interval: tuple[float, float],
    rolling: ModuleType,
) -> OmniscientSolution:
    """求解全知视角的整段联合优化，返回无同时充放电的最优解。

    模型（t 为整段时段的统一编号，共 T 个时段）：

        min  Σ_t p_t g_t
        s.t. g_t + v_t + d_t - s_t = l_t + c_t,
             e_{t+1} = e_t + η_c c_t - d_t / η_d,
             s_t <= v_t + g_t,
             e_0 = 给定初值, E_TERMINAL_MIN <= e_T <= E_TERMINAL_MAX,
             0 <= c_t <= P_c Δt, 0 <= d_t <= P_d Δt,
             E_MIN <= e_t <= E_MAX, g_t >= 0, s_t >= 0.

    s_t 是“剩余电量”（弃光量与未利用购电量之和）。s_t <= v_t + g_t 与
    r_t ∈ [0, v_t]、w_t ∈ [0, g_t]、r_t + w_t = s_t 等价，因此后处理按
    “先弃光”统一取 r_t = min(s_t, v_t)、w_t = s_t - r_t。

    由于 s_t、c_t、d_t 都不出现在目标中，第一阶段 LP 的最优解可能同时充放电；
    第二阶段在费用不超过“下界 + 容差”的前提下最小化 Σ_t (c_t + d_t)，得到同一
    最优费用下的无空转解。该解满足原混合整数模型的充放电互斥约束，因此第一阶段的
    LP 最优值同时就是原混合整数模型的全局最优值。
    """
    start_time = time.time()
    period_count = validate_horizon_inputs(
        load_energy, photovoltaic_energy, price, initial_energy, terminal_interval, rolling
    )
    charge_limit = rolling.CHARGE_LIMIT
    discharge_limit = rolling.DISCHARGE_LIMIT

    offset_grid = 0
    offset_charge = period_count
    offset_discharge = 2 * period_count
    offset_surplus = 3 * period_count
    offset_energy = 4 * period_count
    variable_count = 5 * period_count + 1

    objective = np.zeros(variable_count)
    objective[offset_grid:offset_charge] = price

    lower_bounds = np.zeros(variable_count)
    upper_bounds = np.full(variable_count, np.inf)
    upper_bounds[offset_charge:offset_discharge] = charge_limit
    upper_bounds[offset_discharge:offset_surplus] = discharge_limit
    lower_bounds[offset_energy:offset_energy + period_count + 1] = rolling.E_MIN
    upper_bounds[offset_energy:offset_energy + period_count + 1] = rolling.E_MAX
    lower_bounds[offset_energy] = initial_energy
    upper_bounds[offset_energy] = initial_energy
    lower_bounds[offset_energy + period_count] = terminal_interval[0]
    upper_bounds[offset_energy + period_count] = terminal_interval[1]

    steps = np.arange(period_count)
    balance_rows = np.repeat(steps, 4)
    balance_cols = np.stack([
        offset_grid + steps,
        offset_charge + steps,
        offset_discharge + steps,
        offset_surplus + steps,
    ]).T.ravel()
    balance_data = np.tile(np.array([1.0, -1.0, 1.0, -1.0]), period_count)
    state_rows = np.repeat(period_count + steps, 4)
    state_cols = np.stack([
        offset_energy + steps,
        offset_energy + steps + 1,
        offset_charge + steps,
        offset_discharge + steps,
    ]).T.ravel()
    state_data = np.tile(
        np.array([-1.0, 1.0, -rolling.ETA_CHARGE, 1.0 / rolling.ETA_DISCHARGE]),
        period_count,
    )
    surplus_rows = np.repeat(2 * period_count + steps, 2)
    surplus_cols = np.stack([
        offset_surplus + steps,
        offset_grid + steps,
    ]).T.ravel()
    surplus_data = np.tile(np.array([1.0, -1.0]), period_count)
    matrix = coo_matrix(
        (
            np.concatenate([balance_data, state_data, surplus_data]),
            (
                np.concatenate([balance_rows, state_rows, surplus_rows]),
                np.concatenate([balance_cols, state_cols, surplus_cols]),
            ),
        ),
        shape=(3 * period_count, variable_count),
    ).tocsr()
    equality_matrix = matrix[: 2 * period_count, :]
    equality_rhs = np.concatenate([
        load_energy - photovoltaic_energy,
        np.zeros(period_count),
    ])
    inequality_matrix = matrix[2 * period_count :, :]
    inequality_rhs = photovoltaic_energy
    bounds = np.column_stack((lower_bounds, upper_bounds))
    solver_options = {
        "presolve": True,
        "primal_feasibility_tolerance": 1e-9,
        "dual_feasibility_tolerance": 1e-9,
    }

    relaxed = linprog(
        c=objective,
        A_eq=equality_matrix,
        b_eq=equality_rhs,
        A_ub=inequality_matrix,
        b_ub=inequality_rhs,
        bounds=bounds,
        method="highs-ds",
        options=solver_options,
    )
    if not relaxed.success or relaxed.x is None or not isfinite(float(relaxed.fun)):
        raise RuntimeError(
            f"全知全局 LP 求解失败（{scope_label}）：状态 {relaxed.status}，{relaxed.message}"
        )
    lower_bound = float(relaxed.fun)

    throughput_objective = np.zeros(variable_count)
    throughput_objective[offset_charge:offset_surplus] = 1.0
    cost_row = np.zeros(variable_count)
    cost_row[offset_grid:offset_charge] = price
    selected = None
    tolerance = 0.0
    for candidate in COST_TOLERANCE_CANDIDATES:
        stage_two = linprog(
            c=throughput_objective,
            A_eq=equality_matrix,
            b_eq=equality_rhs,
            A_ub=sparse_vstack([inequality_matrix, cost_row[None, :]], format="csr"),
            b_ub=np.concatenate([inequality_rhs, [lower_bound + candidate]]),
            bounds=bounds,
            method="highs-ds",
            options=solver_options,
        )
        if stage_two.success and stage_two.x is not None:
            selected = stage_two
            tolerance = candidate
            break
    if selected is None or selected.x is None:
        raise RuntimeError(
            f"全知无空转解求解失败（{scope_label}）：状态 {selected.status}，{selected.message}"
        )

    solution = selected.x
    grid_purchase = solution[offset_grid:offset_charge]
    charge = solution[offset_charge:offset_discharge]
    discharge = solution[offset_discharge:offset_surplus]
    surplus = solution[offset_surplus:offset_energy]
    stored_energy = solution[offset_energy:offset_energy + period_count + 1]
    curtailment, unused_purchase = split_surplus(surplus, photovoltaic_energy)

    balance_residual = (
        grid_purchase + photovoltaic_energy + discharge - surplus
        - load_energy - charge
    )
    state_residual = (
        stored_energy[1:] - stored_energy[:-1]
        - rolling.ETA_CHARGE * charge
        + discharge / rolling.ETA_DISCHARGE
    )
    achieved_cost = float(price @ grid_purchase)
    max_surplus_violation = float(np.max(surplus - photovoltaic_energy - grid_purchase))
    max_unused_violation = float(np.max(unused_purchase - grid_purchase))
    max_simultaneous = float(np.max(np.minimum(charge, discharge)))
    max_balance_residual = float(np.max(np.abs(balance_residual)))
    max_state_residual = float(np.max(np.abs(state_residual)))

    if not np.isfinite(solution).all():
        raise RuntimeError(f"全知全局解包含非有限值（{scope_label}）。")
    if max_balance_residual > OMNI_TOLERANCE:
        raise RuntimeError(f"全知全局解电量平衡残差超限（{scope_label}）。")
    if max_state_residual > OMNI_TOLERANCE:
        raise RuntimeError(f"全知全局解状态转移残差超限（{scope_label}）。")
    if max_surplus_violation > OMNI_TOLERANCE:
        raise RuntimeError(f"全知全局解剩余电量超过可用光伏与购电之和（{scope_label}）。")
    if max_unused_violation > OMNI_TOLERANCE:
        raise RuntimeError(f"全知全局解未利用购电量超过当期购电量（{scope_label}）。")
    if max_simultaneous > OMNI_TOLERANCE:
        raise RuntimeError(
            f"全知全局解仍存在同时充放电（{scope_label}），不能宣称为原模型最优解。"
        )
    if np.min(charge) < -OMNI_TOLERANCE or np.min(discharge) < -OMNI_TOLERANCE:
        raise RuntimeError(f"全知全局解出现负充放电（{scope_label}）。")
    if np.max(charge) > charge_limit + OMNI_TOLERANCE:
        raise RuntimeError(f"全知全局解充电量超过功率上限（{scope_label}）。")
    if np.max(discharge) > discharge_limit + OMNI_TOLERANCE:
        raise RuntimeError(f"全知全局解放电量超过功率上限（{scope_label}）。")
    if np.min(stored_energy) < rolling.E_MIN - OMNI_TOLERANCE:
        raise RuntimeError(f"全知全局解储电量低于安全下限（{scope_label}）。")
    if np.max(stored_energy) > rolling.E_MAX + OMNI_TOLERANCE:
        raise RuntimeError(f"全知全局解储电量高于安全上限（{scope_label}）。")
    if achieved_cost > lower_bound + tolerance + OMNI_TOLERANCE:
        raise RuntimeError(f"全知全局解费用超过第一阶段下界的容差范围（{scope_label}）。")

    return OmniscientSolution(
        scope_label=scope_label,
        day_count=period_count // rolling.N_PERIODS,
        period_count=period_count,
        initial_energy=float(initial_energy),
        terminal_interval=(float(terminal_interval[0]), float(terminal_interval[1])),
        lower_bound=lower_bound,
        achieved_cost=achieved_cost,
        cost_tolerance=float(tolerance),
        throughput=float(selected.fun),
        grid_purchase=grid_purchase,
        charge=charge,
        discharge=discharge,
        surplus=surplus,
        curtailment=curtailment,
        unused_purchase=unused_purchase,
        stored_energy=stored_energy,
        max_balance_residual=max_balance_residual,
        max_state_residual=max_state_residual,
        max_surplus_violation=max_surplus_violation,
        max_unused_violation=max_unused_violation,
        max_simultaneous=max_simultaneous,
        min_stored_energy=float(np.min(stored_energy)),
        max_stored_energy=float(np.max(stored_energy)),
        terminal_energy=float(stored_energy[-1]),
        solve_seconds=time.time() - start_time,
    )


def run_perfect_daily_forecast(
    rolling: ModuleType,
    data: Any,
    initial_energy: float,
) -> list[PerfectForecastDay]:
    """L1：把每日 0:00 的预测替换为当天实测，其余策略结构与正式方案一致。"""
    price = np.asarray(data.price, dtype=float)
    # L1 必须与正式方案使用同一个日末软目标系数：主脚本按历史验证在候选集中选择，
    # 当前候选集只有一个元素（λE = max_t p_t），因此这里取该元素的相应倍数。
    candidates = tuple(float(value) for value in rolling.TERMINAL_PENALTY_MULTIPLIERS)
    if len(candidates) != 1:
        raise RuntimeError(
            "本补充脚本假定主脚本的日末惩罚候选集为单元素；"
            "若候选集变化，需同步实现 L1 口径的逐日参数选择。"
        )
    penalty = float(np.max(price)) * candidates[0]
    results: list[PerfectForecastDay] = []
    current_energy = float(initial_energy)
    for day_index in range(rolling.JANUARY_DAYS, len(data.dates)):
        day = data.dates[day_index]
        actual_load = np.asarray(data.actual_load_energy[day_index], dtype=float)
        actual_photovoltaic = np.asarray(
            data.actual_photovoltaic_energy[day_index], dtype=float
        )
        hard_terminal = day == rolling.YEAR_END
        forecast = rolling.Forecast(
            load_energy=actual_load.copy(),
            photovoltaic_energy=actual_photovoltaic.copy(),
            latest_training_date=None,
            used_prior_load=False,
            used_prior_photovoltaic=False,
        )
        # 与主脚本一致：计划阶段在年末采用硬区间，日内执行阶段仍保留日末软目标。
        plan = rolling.solve_validated_plan(
            day, price, forecast, current_energy,
            0.0 if hard_terminal else penalty, hard_terminal,
        )
        replay = rolling.replay_actual_day(
            price, plan, forecast, actual_load, actual_photovoltaic,
            current_energy, penalty,
        )
        rolling.validate_replay(day, plan, replay, actual_load, actual_photovoltaic)
        surplus = np.maximum(
            plan.grid_purchase + actual_photovoltaic + replay.discharge
            + replay.emergency_purchase - actual_load - replay.charge,
            0.0,
        )
        curtailment, unused_purchase = split_surplus(surplus, actual_photovoltaic)
        planned_cost = float(price @ plan.grid_purchase)
        emergency_cost = float(
            rolling.EMERGENCY_PRICE_MULTIPLIER * (price @ replay.emergency_purchase)
        )
        result = PerfectForecastDay(
            day=day,
            initial_energy=current_energy,
            end_energy=float(replay.stored_energy[-1]),
            planned_cost=planned_cost,
            emergency_cost=emergency_cost,
            total_cost=planned_cost + emergency_cost,
            grid_purchase=plan.grid_purchase.copy(),
            charge=replay.charge.copy(),
            discharge=replay.discharge.copy(),
            surplus=surplus,
            curtailment=curtailment,
            unused_purchase=unused_purchase,
            emergency_purchase=replay.emergency_purchase.copy(),
        )
        if rolling.terminal_interval_error(result.end_energy) > rolling.TERMINAL_TOLERANCE:
            raise RuntimeError(f"{day} 完美日预测的年末储电量越界。")
        results.append(result)
        current_energy = result.end_energy
    if len(results) != rolling.OFFICIAL_DAYS:
        raise RuntimeError("完美日预测的正式期天数不正确。")
    return results


def daily_omniscient_slices(
    solution: OmniscientSolution,
    rolling: ModuleType,
) -> list[dict[str, np.ndarray]]:
    """把整段全知解按自然日切片，便于逐日对比。"""
    slices: list[dict[str, np.ndarray]] = []
    for day_offset in range(solution.day_count):
        start = day_offset * rolling.N_PERIODS
        stop = start + rolling.N_PERIODS
        slices.append({
            "grid_purchase": solution.grid_purchase[start:stop],
            "charge": solution.charge[start:stop],
            "discharge": solution.discharge[start:stop],
            "surplus": solution.surplus[start:stop],
            "curtailment": solution.curtailment[start:stop],
            "unused_purchase": solution.unused_purchase[start:stop],
            "stored_energy": solution.stored_energy[start:stop + 1],
        })
    return slices


def build_caliber_rows(
    causal_official: Sequence[CausalDay],
    causal_totals: dict[str, float],
    causal_check: dict[str, float],
    perfect_days: Sequence[PerfectForecastDay],
    omniscient: OmniscientSolution,
    omniscient_fixed_end: OmniscientSolution,
) -> list[Sequence[object]]:
    """生成口径对比表的行。

    L0 行的购电量与弃光量取模型校验 CSV 的权威总量（未经展示舍入累积），
    充放电总量只能由逐时段明细求和，差额在校验表中单独记录。
    """
    causal_cost = causal_check["total_cost"]
    if abs(causal_cost - sum(item.total_cost for item in causal_official)) > 1e-3:
        raise RuntimeError("模型校验 CSV 与每日运行审计的正式期总费用不一致。")
    perfect_cost = sum(item.total_cost for item in perfect_days)
    causal_end = causal_official[-1].end_energy

    def row(
        name: str,
        note: str,
        planned: float,
        emergency: float,
        curtailment: float,
        unused: float,
        charge: float,
        discharge: float,
        cost: float,
        end_energy: float,
        implementable: str,
    ) -> Sequence[object]:
        return (
            name,
            note,
            display_number(planned),
            display_number(emergency),
            display_number(curtailment),
            display_number(unused),
            display_number(charge),
            display_number(discharge),
            display_number(cost),
            display_number(cost - causal_cost),
            display_number((cost - causal_cost) / causal_cost * 100.0),
            display_number(end_energy),
            implementable,
        )

    return [
        row(
            "L0 因果滚动实际执行",
            "问题二正式方案：按日因果预测、0:00 冻结购电量、日内滚动调整储能",
            causal_check["planned_purchase"],
            causal_check["emergency"],
            causal_check["curtailment"],
            causal_check["unused"],
            causal_totals["charge"],
            causal_totals["discharge"],
            causal_cost,
            causal_end,
            "可实施（正式答案）",
        ),
        row(
            "L1 全知日预测同策略",
            "仅把逐日预测替换为当日实测，策略结构、约束与终端条件完全一致",
            float(sum(np.sum(item.grid_purchase) for item in perfect_days)),
            float(sum(np.sum(item.emergency_purchase) for item in perfect_days)),
            float(sum(np.sum(item.curtailment) for item in perfect_days)),
            float(sum(np.sum(item.unused_purchase) for item in perfect_days)),
            float(sum(np.sum(item.charge) for item in perfect_days)),
            float(sum(np.sum(item.discharge) for item in perfect_days)),
            perfect_cost,
            perfect_days[-1].end_energy,
            "不可实施（信息越界）",
        ),
        row(
            "L2 全知全局联合优化（年末区间）",
            "整段联合优化，仅保留年末储电量区间；完美信息下的全局最优，作为理论下界",
            float(np.sum(omniscient.grid_purchase)),
            0.0,
            float(np.sum(omniscient.curtailment)),
            float(np.sum(omniscient.unused_purchase)),
            float(np.sum(omniscient.charge)),
            float(np.sum(omniscient.discharge)),
            omniscient.achieved_cost,
            omniscient.terminal_energy,
            "不可实施（理论下界）",
        ),
        row(
            "L2′ 全知全局联合优化（末态固定）",
            f"与 L2 相同，但年末储电量固定为因果方案实际末态 {causal_end:.6f} kWh",
            float(np.sum(omniscient_fixed_end.grid_purchase)),
            0.0,
            float(np.sum(omniscient_fixed_end.curtailment)),
            float(np.sum(omniscient_fixed_end.unused_purchase)),
            float(np.sum(omniscient_fixed_end.charge)),
            float(np.sum(omniscient_fixed_end.discharge)),
            omniscient_fixed_end.achieved_cost,
            omniscient_fixed_end.terminal_energy,
            "不可实施（理论下界）",
        ),
    ]


def write_caliber_csv(rows: Sequence[Sequence[object]]) -> Path:
    """写出四个口径的总量对比表。"""
    header = (
        "口径", "说明", "计划购电量（kWh）", "紧急购电量（kWh）", "弃光量（kWh）",
        "未利用购电量（kWh）", "充电总量（kWh）", "放电总量（kWh）",
        "购电费（元）", "相对 L0 差额（元）", "相对 L0 差额（%）",
        "年末储电量（kWh）", "可实施性",
    )
    return write_csv(OMNI_CALIBER_FILE, header, rows)


def build_decomposition_rows(
    causal_cost: float,
    perfect_cost: float,
    omniscient: OmniscientSolution,
    omniscient_fixed_end: OmniscientSolution,
    omniscient_year: OmniscientSolution,
    causal_year_cost: float,
) -> list[Sequence[object]]:
    """生成预测信息价值分解表的行。"""
    def relative(value: float, base: float) -> str:
        return display_number(value / base * 100.0)

    return [
        (
            "J0 因果滚动实际总费用",
            "正式方案（L0）的实际总购电费，评价期 334 天",
            display_number(causal_cost),
            "100.000000",
            "元",
            "与实际执行的审计文件一致",
        ),
        (
            "J1 全知日预测同策略总费用",
            "完美日预测但保留 0:00 冻结、日内滚动与日末软目标",
            display_number(perfect_cost),
            relative(perfect_cost, causal_cost),
            "元",
            "不可实施，仅用于分离预测误差",
        ),
        (
            "J2 全知全局最优（理论下界）",
            "整段联合优化的确定性全局最优，年末只保留储电量区间",
            display_number(omniscient.achieved_cost),
            relative(omniscient.achieved_cost, causal_cost),
            "元",
            f"第一阶段 LP 下界 {display_number(omniscient.lower_bound)} 元，"
            f"达成费用与下界之差 {display_number(omniscient.achieved_cost - omniscient.lower_bound)} 元",
        ),
        (
            "Δ预测误差 = J0 − J1",
            "正式方案为预测不确定性付出的费用（相同策略结构下）",
            display_number(causal_cost - perfect_cost),
            relative(causal_cost - perfect_cost, causal_cost),
            "元",
            "占因果费用的大部分，说明差距主要来自预测误差",
        ),
        (
            "Δ逐日结构 = J1 − J2",
            "日末软目标、0:00 冻结购电与逐日决策相对整段联合优化的结构损失",
            display_number(perfect_cost - omniscient.achieved_cost),
            relative(perfect_cost - omniscient.achieved_cost, causal_cost),
            "元",
            "数值很小，说明日末软目标与逐日结构不是主要损失来源",
        ),
        (
            "Δ总差距 = J0 − J2",
            "正式方案相对完美信息全局最优的总差距",
            display_number(causal_cost - omniscient.achieved_cost),
            relative(causal_cost - omniscient.achieved_cost, causal_cost),
            "元",
            "正式方案在预测精度上的理论改进空间上限",
        ),
        (
            "L2′ 固定末态下界",
            "年末储电量固定为因果方案实际末态的全局最优费用",
            display_number(omniscient_fixed_end.achieved_cost),
            relative(omniscient_fixed_end.achieved_cost, causal_cost),
            "元",
            f"与 L2 相差 {display_number(omniscient_fixed_end.achieved_cost - omniscient.achieved_cost)} 元，"
            "说明年末区间内的末态选择对结论影响很小",
        ),
        (
            "J365 全年理论下界",
            "2025-01-01 至 12-31 整段联合优化的全局最优费用",
            display_number(omniscient_year.achieved_cost),
            relative(omniscient_year.achieved_cost, causal_year_cost),
            "元",
            f"同口径因果全年费用 {display_number(causal_year_cost)} 元，"
            f"差距 {display_number(causal_year_cost - omniscient_year.achieved_cost)} 元",
        ),
    ]


def write_decomposition_csv(rows: Sequence[Sequence[object]]) -> Path:
    """写出预测信息价值分解表。"""
    header = ("项目", "定义", "数值", "占基准费用比例（%）", "单位", "说明")
    return write_csv(OMNI_DECOMPOSITION_FILE, header, rows)


def write_daily_csv(
    rolling: ModuleType,
    data: Any,
    causal_daily: dict[date, CausalDay],
    perfect_days: Sequence[PerfectForecastDay],
    omniscient: OmniscientSolution,
) -> Path:
    """写出逐日对比表：因果实际、完美日预测与全知全局分摊。"""
    slices = daily_omniscient_slices(omniscient, rolling)
    official_dates = list(data.dates[rolling.JANUARY_DAYS:])
    if len(official_dates) != len(perfect_days) or len(official_dates) != len(slices):
        raise RuntimeError("逐日对比表的数据长度不一致。")

    causal_cumulative = 0.0
    perfect_cumulative = 0.0
    omniscient_cumulative = 0.0
    rows: list[Sequence[object]] = []
    for day, perfect, omniscient_slice in zip(official_dates, perfect_days, slices):
        causal = causal_daily[day]
        if perfect.day != day:
            raise RuntimeError("完美日预测的日期顺序与正式期不一致。")
        omniscient_cost = float(data.price @ omniscient_slice["grid_purchase"])
        causal_cumulative += causal.total_cost
        perfect_cumulative += perfect.total_cost
        omniscient_cumulative += omniscient_cost
        rows.append((
            day.isoformat(),
            display_number(causal.total_cost),
            display_number(perfect.total_cost),
            display_number(omniscient_cost),
            display_number(causal.end_energy),
            display_number(perfect.end_energy),
            display_number(float(omniscient_slice["stored_energy"][-1])),
            display_number(causal.emergency_energy),
            display_number(float(np.sum(perfect.emergency_purchase))),
            display_number(causal_cumulative),
            display_number(perfect_cumulative),
            display_number(omniscient_cumulative),
        ))
    return write_csv(
        OMNI_DAILY_FILE,
        (
            "日期", "L0 因果实际总购电费（元）", "L1 完美日预测总购电费（元）",
            "L2 全知全局购电费（元）", "L0 实际日末储电量（kWh）",
            "L1 日末储电量（kWh）", "L2 日末储电量（kWh）",
            "L0 紧急购电量（kWh）", "L1 紧急购电量（kWh）",
            "L0 累计购电费（元）", "L1 累计购电费（元）", "L2 累计购电费（元）",
        ),
        rows,
    )


def write_detail_csv(
    rolling: ModuleType,
    data: Any,
    omniscient: OmniscientSolution,
) -> Path:
    """写出全知全局最优解的 334 天逐时段计划与储电量轨迹。"""
    rows: list[Sequence[object]] = []
    period = 0
    for day_index in range(rolling.JANUARY_DAYS, len(data.dates)):
        day = data.dates[day_index]
        for period_index, label in enumerate(data.interval_labels):
            rows.append((
                day.isoformat(),
                label,
                display_number(data.price[period_index]),
                display_number(data.actual_load_energy[day_index][period_index]),
                display_number(data.actual_photovoltaic_energy[day_index][period_index]),
                display_number(omniscient.grid_purchase[period]),
                display_number(omniscient.charge[period]),
                display_number(omniscient.discharge[period]),
                display_number(omniscient.curtailment[period]),
                display_number(omniscient.unused_purchase[period]),
                display_number(omniscient.stored_energy[period]),
                display_number(omniscient.stored_energy[period + 1]),
                display_number(data.price[period_index] * omniscient.grid_purchase[period]),
            ))
            period += 1
    if period != omniscient.period_count:
        raise RuntimeError("全知明细表的时段数与求解规模不一致。")
    return write_csv(
        OMNI_DETAIL_FILE,
        (
            "日期", "时间段", "电价（元/kWh）", "实际负载电量（kWh）",
            "实际光伏电量（kWh）", "全知购电量（kWh）", "全知充电量（kWh）",
            "全知放电量（kWh）", "全知弃光量（kWh）", "全知未利用购电量（kWh）",
            "时段初储电量（kWh）", "时段末储电量（kWh）", "时段购电费（元）",
        ),
        rows,
    )


def verify_lower_bound_embedding(
    rolling: ModuleType,
    data: Any,
    omniscient: OmniscientSolution,
    causal_cost: float,
) -> list[Sequence[object]]:
    """按第 2.4 节命题把 L0 的实际执行轨迹嵌入 L2，并在真实数据上核对下界。

    嵌入点取 g' = g^plan + h、c' = c、d' = d、s' = r + w、e' = e。逐项校验
    平衡式、状态转移与 s' <= v + g'，并核对下界链条
    J^omni <= Σ p g' <= J^real。数值来自逐时段明细，含 6 位小数的展示舍入，
    因此残差容差取 1e-3。
    """
    required = (
        "日期", "时段初储电量（kWh）", "时段末储电量（kWh）",
        "电价（元/kWh）", "实际负载电量（kWh）", "实际光伏电量（kWh）",
        "计划购电量（kWh）", "实际充电量（kWh）", "实际放电量（kWh）",
        "紧急购电量（kWh）", "实际弃光量（kWh）", "未利用计划购电量（kWh）",
    )
    if not DETAIL_FILE.is_file():
        raise RuntimeError(f"未找到 {DETAIL_FILE}。")
    columns: dict[str, list[float]] = {name: [] for name in required if name != "日期"}
    observed_dates: list[date] = []
    with DETAIL_FILE.open(encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None or any(name not in reader.fieldnames for name in required):
            raise RuntimeError(f"{DETAIL_FILE.name} 缺少下界嵌入校验所需列。")
        first_initial: float | None = None
        for row in reader:
            observed_dates.append(date.fromisoformat(row["日期"]))
            if first_initial is None:
                first_initial = float(row["时段初储电量（kWh）"])
            for name in columns:
                columns[name].append(float(row[name]))
    expected_dates = list(data.dates[rolling.JANUARY_DAYS:])
    day_count = len(columns["时段末储电量（kWh）"]) // rolling.N_PERIODS
    if (
        first_initial is None
        or day_count * rolling.N_PERIODS != len(columns["时段末储电量（kWh）"])
        or observed_dates[::rolling.N_PERIODS] != expected_dates[:day_count]
    ):
        raise RuntimeError("逐时段明细的日期顺序或时段数与正式期不一致。")

    price = np.asarray(columns["电价（元/kWh）"])
    load = np.asarray(columns["实际负载电量（kWh）"])
    photovoltaic = np.asarray(columns["实际光伏电量（kWh）"])
    grid = (
        np.asarray(columns["计划购电量（kWh）"])
        + np.asarray(columns["紧急购电量（kWh）"])
    )
    charge = np.asarray(columns["实际充电量（kWh）"])
    discharge = np.asarray(columns["实际放电量（kWh）"])
    surplus = (
        np.asarray(columns["实际弃光量（kWh）"])
        + np.asarray(columns["未利用计划购电量（kWh）"])
    )
    stored = np.concatenate([
        np.array([float(first_initial)]),
        np.asarray(columns["时段末储电量（kWh）"]),
    ])

    balance_residual = grid + photovoltaic + discharge - surplus - load - charge
    state_residual = (
        stored[1:] - stored[:-1]
        - rolling.ETA_CHARGE * charge
        + discharge / rolling.ETA_DISCHARGE
    )
    surplus_margin = surplus - photovoltaic - grid
    embedded_cost = float(price @ grid)
    tolerance = 1e-3

    def conclusion(value: float, limit: float, meaningful: bool = True) -> str:
        if not meaningful:
            return "参考"
        return "通过" if abs(value) <= limit else "失败"

    return [
        ("下界命题", "嵌入点最大电量平衡残差",
         display_number(float(np.max(np.abs(balance_residual)))), "kWh",
         conclusion(float(np.max(np.abs(balance_residual))), tolerance)),
        ("下界命题", "嵌入点最大状态转移残差",
         display_number(float(np.max(np.abs(state_residual)))), "kWh",
         conclusion(float(np.max(np.abs(state_residual))), tolerance)),
        ("下界命题", "嵌入点剩余电量超过可用光伏与购电之和的最大值",
         display_number(float(np.max(surplus_margin))), "kWh",
         "通过" if float(np.max(surplus_margin)) <= tolerance else "失败"),
        ("下界命题", "嵌入点购电费用（Σ p(g_plan+h)）", display_number(embedded_cost), "元", "参考"),
        ("下界命题", "嵌入点费用减 L2 下界", display_number(embedded_cost - omniscient.lower_bound), "元",
         "通过" if embedded_cost >= omniscient.lower_bound - tolerance else "失败"),
        ("下界命题", "L0 实际费用减嵌入点费用（紧急电价 5 倍）",
         display_number(causal_cost - embedded_cost), "元",
         "通过" if causal_cost >= embedded_cost - tolerance else "失败"),
        ("下界命题", "嵌入点年末储电量", display_number(float(stored[-1])), "kWh",
         "通过" if rolling.terminal_interval_error(float(stored[-1])) <= rolling.TERMINAL_TOLERANCE else "失败"),
    ]


def build_check_rows(
    rolling: ModuleType,
    causal_cost: float,
    causal_totals: dict[str, float],
    causal_check: dict[str, float],
    causal_detail_cost: float,
    perfect_days: Sequence[PerfectForecastDay],
    perfect_seconds: float,
    omniscient: OmniscientSolution,
    omniscient_fixed_end: OmniscientSolution,
    omniscient_year: OmniscientSolution,
    embedding_rows: Sequence[Sequence[object]],
) -> list[Sequence[object]]:
    """汇总全知视角的求解精度、约束残差与差距分解。"""
    perfect_cost = sum(item.total_cost for item in perfect_days)
    perfect_emergency = float(sum(np.sum(item.emergency_purchase) for item in perfect_days))
    return [
        ("L2 求解", "第一阶段 LP 最优值（严格下界）", display_number(omniscient.lower_bound), "元", "参考"),
        ("L2 求解", "第二阶段无空转可行解费用", display_number(omniscient.achieved_cost), "元", "通过"),
        ("L2 求解", "第二阶段费用上浮容差", display_number(omniscient.cost_tolerance), "元", "参考"),
        ("L2 求解", "达成费用减下界", display_number(omniscient.achieved_cost - omniscient.lower_bound), "元",
         "通过" if omniscient.achieved_cost - omniscient.lower_bound <= omniscient.cost_tolerance + OMNI_TOLERANCE else "失败"),
        ("L2 求解", "第二阶段最小充放电总吞吐", display_number(omniscient.throughput), "kWh", "参考"),
        ("L2 求解", "三个整段问题求解用时合计",
         display_number(omniscient.solve_seconds + omniscient_fixed_end.solve_seconds + omniscient_year.solve_seconds),
         "s", "参考"),
        ("L2 约束", "最大同时充放电量", display_number(omniscient.max_simultaneous), "kWh", "通过"),
        ("L2 约束", "最大电量平衡残差", display_number(omniscient.max_balance_residual), "kWh", "通过"),
        ("L2 约束", "最大状态转移残差", display_number(omniscient.max_state_residual), "kWh", "通过"),
        ("L2 约束", "剩余电量超过可用光伏与购电之和的最大值", display_number(omniscient.max_surplus_violation), "kWh", "通过"),
        ("L2 约束", "未利用购电量超过当期购电量的最大值", display_number(omniscient.max_unused_violation), "kWh", "通过"),
        ("L2 边界", "全知储电量最小值", display_number(omniscient.min_stored_energy), "kWh", "通过"),
        ("L2 边界", "全知储电量最大值", display_number(omniscient.max_stored_energy), "kWh", "通过"),
        ("L2 边界", "全知年末储电量", display_number(omniscient.terminal_energy), "kWh",
         "通过" if rolling.terminal_interval_error(omniscient.terminal_energy) <= rolling.TERMINAL_TOLERANCE else "失败"),
        ("L2 数量", "全知计划购电总量", display_number(float(np.sum(omniscient.grid_purchase))), "kWh", "参考"),
        ("L2 数量", "全知充电总量", display_number(float(np.sum(omniscient.charge))), "kWh", "参考"),
        ("L2 数量", "全知放电总量", display_number(float(np.sum(omniscient.discharge))), "kWh", "参考"),
        ("L2 数量", "全知弃光总量", display_number(float(np.sum(omniscient.curtailment))), "kWh", "参考"),
        ("L2 数量", "全知未利用购电总量", display_number(float(np.sum(omniscient.unused_purchase))), "kWh", "参考"),
        ("L2 稳健性", "L2′ 固定末态下界费用", display_number(omniscient_fixed_end.achieved_cost), "元", "参考"),
        ("L2 稳健性", "L2′ 减 L2", display_number(omniscient_fixed_end.achieved_cost - omniscient.achieved_cost), "元", "参考"),
        ("L2 稳健性", "全年 365 天下界费用", display_number(omniscient_year.achieved_cost), "元", "参考"),
        ("L1 校验", "完美日预测总费用", display_number(perfect_cost), "元", "参考"),
        ("L1 校验", "完美日预测紧急购电量", display_number(perfect_emergency), "kWh",
         "通过" if abs(perfect_emergency) <= rolling.BALANCE_TOLERANCE else "参考"),
        ("L1 校验", "完美日预测末态储电量", display_number(perfect_days[-1].end_energy), "kWh", "通过"),
        ("L1 校验", "完美日预测求解失败次数", "0", "次", "通过"),
        ("L1 校验", "完美日预测用时", display_number(perfect_seconds), "s", "参考"),
        ("差距", "Δ预测误差 = J0 − J1", display_number(causal_cost - perfect_cost), "元", "参考"),
        ("差距", "Δ逐日结构 = J1 − J2", display_number(perfect_cost - omniscient.achieved_cost), "元", "参考"),
        ("差距", "Δ总差距 = J0 − J2", display_number(causal_cost - omniscient.achieved_cost), "元", "参考"),
        ("差距", "Δ总差距占 J0 比例", display_number((causal_cost - omniscient.achieved_cost) / causal_cost), "比例", "参考"),
        ("交叉校验", "L0 明细逐时段费用求和", display_number(causal_detail_cost), "元", "参考"),
        ("交叉校验", "L0 明细费用求和减审计费用", display_number(causal_detail_cost - causal_cost), "元",
         "通过" if abs(causal_detail_cost - causal_cost) <= 1.0 else "参考"),
        ("交叉校验", "L0 计划购电总量（明细求和，含展示舍入）",
         display_number(causal_totals["planned_purchase"]), "kWh", "参考"),
        ("交叉校验", "L0 计划购电总量（明细求和减模型校验）",
         display_number(causal_totals["planned_purchase"] - causal_check["planned_purchase"]), "kWh", "参考"),
        ("交叉校验", "L0 弃光总量（明细求和减模型校验）",
         display_number(causal_totals["curtailment"] - causal_check["curtailment"]), "kWh", "参考"),
        ("交叉校验", "L0 未利用购电总量（明细求和减模型校验）",
         display_number(causal_totals["unused"] - causal_check["unused"]), "kWh", "参考"),
        *embedding_rows,
    ]


def write_check_csv(rows: Sequence[Sequence[object]]) -> Path:
    """写出全知视角校验表。"""
    return write_csv(OMNI_CHECK_FILE, ("类别", "校验项", "数值", "单位", "结论"), rows)


def main() -> None:
    """完成全知视角三个口径的求解、对比与校验。"""
    check_environment()
    rolling = load_rolling_module()
    data = rolling.load_model_data()
    causal_daily, causal_official = read_causal_audit(rolling.JANUARY_DAYS)
    causal_totals, causal_detail_cost = read_causal_detail()
    causal_check = read_causal_check_totals()
    causal_cost = causal_check["total_cost"]
    causal_year_cost = sum(item.total_cost for item in causal_daily.values())

    official_initial_energy = causal_official[0].initial_energy
    if abs(official_initial_energy - rolling.E_INITIAL) > 1e-6:
        raise RuntimeError(
            f"正式期初始储电量 {official_initial_energy:.6f} kWh 与主脚本约定不一致。"
        )
    causal_end_energy = causal_official[-1].end_energy
    official_dates = data.dates[rolling.JANUARY_DAYS:]
    official_load = data.actual_load_energy[rolling.JANUARY_DAYS:].ravel()
    official_photovoltaic = data.actual_photovoltaic_energy[rolling.JANUARY_DAYS:].ravel()
    official_price = np.tile(np.asarray(data.price, dtype=float), len(official_dates))

    omniscient_official = solve_omniscient_horizon(
        "正式期 334 天（年末区间）",
        official_load, official_photovoltaic, official_price,
        official_initial_energy,
        (rolling.E_TERMINAL_MIN, rolling.E_TERMINAL_MAX),
        rolling,
    )
    omniscient_fixed_end = solve_omniscient_horizon(
        "正式期 334 天（末态固定为因果实际末态）",
        official_load, official_photovoltaic, official_price,
        official_initial_energy,
        (causal_end_energy, causal_end_energy),
        rolling,
    )
    omniscient_year = solve_omniscient_horizon(
        "全年 365 天（年末区间）",
        data.actual_load_energy.ravel(),
        data.actual_photovoltaic_energy.ravel(),
        np.tile(np.asarray(data.price, dtype=float), len(data.dates)),
        rolling.E_INITIAL,
        (rolling.E_TERMINAL_MIN, rolling.E_TERMINAL_MAX),
        rolling,
    )

    perfect_start = time.time()
    perfect_days = run_perfect_daily_forecast(rolling, data, official_initial_energy)
    perfect_seconds = time.time() - perfect_start
    perfect_cost = sum(item.total_cost for item in perfect_days)

    OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    embedding_rows = verify_lower_bound_embedding(
        rolling, data, omniscient_official, causal_cost
    )
    outputs = [
        write_caliber_csv(build_caliber_rows(
            causal_official, causal_totals, causal_check, perfect_days,
            omniscient_official, omniscient_fixed_end,
        )),
        write_decomposition_csv(build_decomposition_rows(
            causal_cost, perfect_cost, omniscient_official,
            omniscient_fixed_end, omniscient_year, causal_year_cost,
        )),
        write_daily_csv(rolling, data, causal_daily, perfect_days, omniscient_official),
        write_detail_csv(rolling, data, omniscient_official),
        write_check_csv(build_check_rows(
            rolling, causal_cost, causal_totals, causal_check, causal_detail_cost,
            perfect_days, perfect_seconds, omniscient_official,
            omniscient_fixed_end, omniscient_year, embedding_rows,
        )),
    ]

    total_gap = causal_cost - omniscient_official.achieved_cost
    print(f"L0 因果滚动实际总费用：{causal_cost:.6f} 元")
    print(f"L1 全知日预测同策略总费用：{perfect_cost:.6f} 元")
    print(
        f"L2 全知全局理论下界：{omniscient_official.lower_bound:.6f} 元，"
        f"达成费用：{omniscient_official.achieved_cost:.6f} 元"
    )
    print(
        f"Δ预测误差 = {causal_cost - perfect_cost:.6f} 元，"
        f"Δ逐日结构 = {perfect_cost - omniscient_official.achieved_cost:.6f} 元"
    )
    print(f"Δ总差距 = {total_gap:.6f} 元（{total_gap / causal_cost * 100:.6f}%）")
    for path in outputs:
        print(f"已保存：{path.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
