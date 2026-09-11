"""为什么所有 lambda 结果相同？用 extreme 惩罚测试终端惩罚是否真的进入目标函数。"""
import sys, importlib.util
from pathlib import Path
import numpy as np

P = Path("/Users/mistvillion/Documents/GitHub Repositories/Mathematical-Modelling/2026C")
spec = importlib.util.spec_from_file_location("q2", P / "scripts_tables/2_rolling_optimization.py")
q2 = importlib.util.module_from_spec(spec); sys.modules["q2"] = q2; spec.loader.exec_module(q2)

data = q2.load_model_data()
engine = q2.ForecastEngine("cpu")

def forecast_for(day_index, load_decay=10.0, pv_decay=7.0):
    day = data.dates[day_index]
    hist = data.dates[:day_index]
    lc, pc = engine.candidates(day, hist, data.actual_load_energy[:day_index],
                               data.actual_photovoltaic_energy[:day_index],
                               data.cold_start_load_energy, data.cold_start_photovoltaic_energy)
    return q2.Forecast(lc[q2.DECAY_DAY_CANDIDATES.index(load_decay)],
                       pc[q2.DECAY_DAY_CANDIDATES.index(pv_decay)],
                       engine.latest_training_date(day, hist, "marginal"), False, False)

for day_index in (31, 300, 364):
    fc = forecast_for(day_index)
    day = data.dates[day_index]
    print(f"\n=== {day} (soft terminal, e0=6666.47) ===")
    for lam in (0.0, 1e-9, 0.05, 0.6976, 1.3952, 6.976, 100.0):
        plan = q2.solve_daily_plan(data.price, fc, 6666.471759, lam, False)
        print(f"  lambda={lam:>10.6f} obj={plan.objective_value:12.4f} sumGrid={plan.grid_purchase.sum():12.4f} "
              f"e_N={plan.stored_energy[-1]:10.6f} z+={plan.positive_terminal_deviation:.6f} "
              f"z-={plan.negative_terminal_deviation:.6f}")
