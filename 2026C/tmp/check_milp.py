"""交叉验证：LP 路径解 vs 完整 MILP 解是否同值；计划是否出现购电+弃光/充放矛盾。"""
import sys, importlib.util
from pathlib import Path
import numpy as np
from scipy.optimize import milp
import pandas as pd

P = Path("/Users/mistvillion/Documents/GitHub Repositories/Mathematical-Modelling/2026C")
spec = importlib.util.spec_from_file_location("q2", P / "scripts_tables/2_rolling_optimization.py")
q2 = importlib.util.module_from_spec(spec); sys.modules["q2"] = q2; spec.loader.exec_module(q2)
data = q2.load_model_data()

# ---- 计划层面的"购电同时弃光"检查 ----
det = pd.read_csv(P / "outputs/tables/2_cpu_marginal/2_滚动预测与调度明细.csv", encoding="utf-8-sig")
gp = det["计划购电量（kWh）"].to_numpy(float)
cp = det["计划充电量（kWh）"].to_numpy(float)
dp = det["计划放电量（kWh）"].to_numpy(float)
fl = det["预测负载电量（kWh）"].to_numpy(float)
fp = det["预测光伏电量（kWh）"].to_numpy(float)
curt = gp + fp + dp - fl - cp
print("=== 计划内部矛盾检查 ===")
print(f"计划弃光>1e-6 且 计划购电>1e-6 的时段: {int(((curt>1e-6)&(gp>1e-6)).sum())}")
print(f"计划弃光>1e-6 且 计划充电>1e-6 的时段: {int(((curt>1e-6)&(cp>1e-6)).sum())}")
print(f"计划弃光>1e-6 且 计划放电>1e-6 的时段: {int(((curt>1e-6)&(dp>1e-6)).sum())}")
print(f"计划购电>1e-6 且 计划放电>1e-6 的时段: {int(((gp>1e-6)&(dp>1e-6)).sum())}")
print(f"计划购电>1e-6 且 计划弃光>1e-6 且 计划充电>1e-6: {int(((gp>1e-6)&(curt>1e-6)&(cp>1e-6)).sum())}")

# ---- LP 路径 vs 完整 MILP ----
engine = q2.ForecastEngine("cpu")
print("\n=== LP 路径 vs 强制完整 MILP（目标值应相同）===")
for day_index in (31, 60, 150, 250, 364):
    day = data.dates[day_index]
    hist = data.dates[:day_index]
    lc, pc = engine.candidates(day, hist, data.actual_load_energy[:day_index],
                               data.actual_photovoltaic_energy[:day_index],
                               data.cold_start_load_energy, data.cold_start_photovoltaic_energy)
    fc = q2.Forecast(lc[q2.DECAY_DAY_CANDIDATES.index(5.0)], pc[q2.DECAY_DAY_CANDIDATES.index(5.0)],
                     engine.latest_training_date(day, hist, "marginal"), False, False)
    for e0 in (6000.0, 6500.0):
        lp = q2.solve_daily_plan(data.price, fc, e0, 1.3952, day == q2.YEAR_END, force_milp=False)
        mip = q2.solve_daily_plan(data.price, fc, e0, 1.3952, day == q2.YEAR_END, force_milp=True)
        same = abs(lp.objective_value - mip.objective_value) < 1e-6
        print(f"  {day} e0={e0}: LP[{lp.solver_kind[:2]}] obj={lp.objective_value:.9f} e_N={lp.stored_energy[-1]:.6f} | "
              f"MILP obj={mip.objective_value:.9f} e_N={mip.stored_energy[-1]:.6f} | 一致={same} "
              f"充电量差={np.max(np.abs(lp.charge-mip.charge)):.2e}")
