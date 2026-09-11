"""检验不同终端惩罚候选对同日计划是否真的产生不同结果。"""
import sys, importlib.util
from pathlib import Path
import numpy as np

P = Path("/Users/mistvillion/Documents/GitHub Repositories/Mathematical-Modelling/2026C")
spec = importlib.util.spec_from_file_location("q2", P / "scripts_tables/2_rolling_optimization.py")
q2 = importlib.util.module_from_spec(spec)
sys.modules["q2"] = q2
spec.loader.exec_module(q2)

data = q2.load_model_data()
engine = q2.ForecastEngine("cpu")
penalties = np.max(data.price) * np.asarray(q2.TERMINAL_PENALTY_MULTIPLIERS)
print("penalties:", penalties, "max price:", np.max(data.price))

for day_index in (31, 120, 300, 364):
    day = data.dates[day_index]
    history_dates = data.dates[:day_index]
    load_candidates, pv_candidates = engine.candidates(
        day, history_dates, data.actual_load_energy[:day_index],
        data.actual_photovoltaic_energy[:day_index],
        data.cold_start_load_energy, data.cold_start_photovoltaic_energy)
    load_index = q2.DECAY_DAY_CANDIDATES.index(10.0)
    pv_index = q2.DECAY_DAY_CANDIDATES.index(7.0)
    fc = q2.Forecast(load_candidates[load_index], pv_candidates[pv_index],
                     engine.latest_training_date(day, history_dates, "marginal"), False, False)
    # 初始储电量：逐日模拟成本高，这里用 6000 与 6666 各试一次
    for e0 in (6000.0, 6666.471759):
        print(f"\n--- day={day} e0={e0} ---")
        results = []
        for pen in penalties:
            plan = q2.solve_daily_plan(data.price, fc, e0, float(pen), day == q2.YEAR_END)
            results.append(plan)
        base = results[0]
        for pen, plan in zip(penalties, results):
            print(f"  lambda={pen:8.4f} obj={plan.objective_value:12.6f} "
                  f"sumGrid={plan.grid_purchase.sum():12.6f} e_N={plan.stored_energy[-1]:10.6f} "
                  f"Δgrid_vs_c0={plan.grid_purchase.sum()-base.grid_purchase.sum():+.6f} "
                  f"max|Δgrid_feasible|={np.max(np.abs(plan.grid_purchase-base.grid_purchase)):.6f}")
