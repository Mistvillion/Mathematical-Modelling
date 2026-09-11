"""对照实验：把终端惩罚候选扩到阈值以下，观察年初年末误差与计划日末储电量。"""
import sys, importlib.util, json
from pathlib import Path
from types import SimpleNamespace
import numpy as np

P = Path("/Users/mistvillion/Documents/GitHub Repositories/Mathematical-Modelling/2026C")
spec = importlib.util.spec_from_file_location("q2", P / "scripts_tables/2_rolling_optimization.py")
q2 = importlib.util.module_from_spec(spec); sys.modules["q2"] = q2; spec.loader.exec_module(q2)
data = q2.load_model_data()
engine = q2.ForecastEngine("cpu")
maxp = float(np.max(data.price))
print("max price =", maxp)

for label, mult in (("lambda=0.0698 (0.05x)", 0.05), ("lambda=0.3488 (0.25x)", 0.25), ("lambda=1.3952 (1.0x, 现用)", 1.0)):
    q2.TERMINAL_PENALTY_MULTIPLIERS = (mult, mult)   # 两候选同值，保持 default_index=1 可用
    q2.OUTPUT_TABLE_DIR = P / "tmp" / f"exp_{mult}"
    args = SimpleNamespace(device="cpu", workers=1, forecast_mode="marginal",
                           run_name="exp", resume=False, force_milp=False,
                           log_every=10**9, self_check=False)
    rolling = q2.solve_rolling_model(data, engine, args)
    err = q2.validate_rolling_result(data, rolling)
    official = rolling.official_days
    total = sum(d.realized_total_cost for d in official)
    em = sum(d.emergency_purchase_cost for d in official)
    planned_end = np.array([d.plan.stored_energy[-1] for d in official])
    actual_end = np.array([d.replay.stored_energy[-1] for d in official])
    print(f"{label:>26s}: 年末实际={actual_end[-1]:10.4f} 误差={err:9.4f} kWh | 总费={total:12.2f} 紧急费={em:10.2f} "
          f"| 计划日末 min={planned_end.min():9.3f} max={planned_end.max():9.3f} mean={planned_end.mean():9.3f} "
          f"!=6000的天数={int((np.abs(planned_end-6000)>1e-3).sum())}")
