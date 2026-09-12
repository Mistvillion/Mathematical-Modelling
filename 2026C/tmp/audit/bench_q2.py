"""Quality benchmarks for the Q2 rolling model.

Reuses the project's forecast/plan/replay primitives so the comparison is
apples-to-apples, but drives them from an external rolling loop so that the
forecast source, the risk quantiles and the terminal-penalty candidates can be
varied.  Nothing here writes into outputs/.

Usage:  python tmp/audit/bench_q2.py <variant>
variants: user, q50, q70, q90, cold, perfect, pen0, pen05, pen2, pen5
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

P = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "q2", Path(__file__).resolve().parent / "q2_head_frozen.py"
)
q2 = importlib.util.module_from_spec(spec)
sys.modules["q2"] = q2
spec.loader.exec_module(q2)
# the frozen copy lives under tmp/, so re-point the attachment paths
q2.PROJECT_DIR = P
q2.ATTACHMENT_DIR = P / "CUMCM 2026 C题" / "附件"
q2.PRICE_AND_PRIOR_FILE = q2.ATTACHMENT_DIR / "附件1.xlsx"
q2.ACTUAL_DATA_FILE = q2.ATTACHMENT_DIR / "附件2.xlsx"
q2.RESULT_TEMPLATE_FILE = q2.ATTACHMENT_DIR / "附件5" / "result2.xlsx"

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "user"
NEG_INF = -np.inf

CONFIG = {
    "user":    dict(q_load=0.8, q_pv=0.2, mults=(0.5, 1.0, 2.0, 5.0), source="hist"),
    "q50":     dict(q_load=0.5, q_pv=0.5, mults=(0.5, 1.0, 2.0, 5.0), source="hist"),
    "q70":     dict(q_load=0.7, q_pv=0.3, mults=(0.5, 1.0, 2.0, 5.0), source="hist"),
    "q90":     dict(q_load=0.9, q_pv=0.1, mults=(0.5, 1.0, 2.0, 5.0), source="hist"),
    "cold":    dict(q_load=0.8, q_pv=0.2, mults=(0.5, 1.0, 2.0, 5.0), source="cold"),
    "perfect": dict(q_load=0.8, q_pv=0.2, mults=(0.5, 1.0, 2.0, 5.0), source="actual"),
    "pfload":  dict(q_load=0.8, q_pv=0.2, mults=(0.5, 1.0, 2.0, 5.0), source="perfect_load"),
    "rtbat":   dict(q_load=0.8, q_pv=0.2, mults=(1.0,), source="hist", rt_battery=True),
    "pfpv":    dict(q_load=0.8, q_pv=0.2, mults=(0.5, 1.0, 2.0, 5.0), source="perfect_pv"),
    # penalty sensitivity: only one candidate -> deterministic lambda_E
    "pen0":    dict(q_load=0.8, q_pv=0.2, mults=(0.0,), source="hist"),
    "pen02":   dict(q_load=0.8, q_pv=0.2, mults=(0.2,), source="hist"),
    "pen05":   dict(q_load=0.8, q_pv=0.2, mults=(0.5,), source="hist"),
    "pen2":    dict(q_load=0.8, q_pv=0.2, mults=(2.0,), source="hist"),
    "pen5":    dict(q_load=0.8, q_pv=0.2, mults=(5.0,), source="hist"),
}
cfg = CONFIG[VARIANT]
print(f"### variant={VARIANT} cfg={cfg}", flush=True)

N = q2.N_PERIODS; LIM = 5000 / 6  # noqa: E741
ETA_C = q2.ETA_CHARGE; ETA_D = q2.ETA_DISCHARGE; E_REF = q2.E_REFERENCE
E_MIN = q2.E_MIN; E_MAX = q2.E_MAX; PEN = q2.EMERGENCY_PRICE_MULTIPLIER
data = q2.load_model_data()
LAM = float(np.max(data.price))
engine = q2.ForecastEngine()
price = data.price
penalties = np.max(price) * np.asarray(cfg["mults"])
terminal_value = float(np.max(price) / q2.ETA_CHARGE)
default_decay = q2.DECAY_DAY_CANDIDATES.index(q2.DEFAULT_DECAY_DAYS)
default_penalty = (
    cfg["mults"].index(1.0) if 1.0 in cfg["mults"] else 0
)

def rt_dispatch(g_frozen, la, va, e0, hard_terminal):
    """Real-time optimal battery dispatch with the purchase contract frozen."""
    from scipy.optimize import linprog as _lp
    from scipy.sparse import lil_matrix as _lil
    nv = 6 * N + 3
    oo = dict(h=0, c=N, d=2 * N, r=3 * N, w=4 * N, e=5 * N)
    zp, zm = 6 * N + 1, 6 * N + 2
    cv = np.zeros(nv)
    cv[oo["h"]:oo["h"] + N] = PEN * price
    cv[zp] = cv[zm] = LAM
    lb = np.zeros(nv)
    ub = np.full(nv, np.inf)
    ub[oo["c"]:oo["c"] + N] = LIM
    ub[oo["d"]:oo["d"] + N] = LIM
    ub[oo["r"]:oo["r"] + N] = va
    ub[oo["w"]:oo["w"] + N] = g_frozen
    lb[oo["e"]:oo["e"] + N + 1] = E_MIN
    ub[oo["e"]:oo["e"] + N + 1] = E_MAX
    lb[oo["e"]] = ub[oo["e"]] = e0
    AA = _lil((2 * N + 1, nv))
    lo = np.full(2 * N + 1, -np.inf)
    hi = np.full(2 * N + 1, np.inf)
    for t in range(N):
        AA[t, oo["h"] + t] = 1
        AA[t, oo["c"] + t] = -1
        AA[t, oo["d"] + t] = 1
        AA[t, oo["r"] + t] = -1
        AA[t, oo["w"] + t] = -1
        lo[t] = hi[t] = la[t] - va[t] - g_frozen[t]
        r = N + t
        AA[r, oo["c"] + t] = -ETA_C
        AA[r, oo["d"] + t] = 1 / ETA_D
        AA[r, oo["e"] + t] = -1
        AA[r, oo["e"] + t + 1] = 1
        lo[r] = hi[r] = 0.0
    AA[2 * N, oo["e"] + N] = 1.0
    AA[2 * N, zp] = -1.0
    AA[2 * N, zm] = 1.0
    lo[2 * N] = hi[2 * N] = E_REF
    if hard_terminal:
        lb[oo["e"] + N] = 5400.0
        ub[oo["e"] + N] = 6600.0
        ub[zp] = ub[zm] = 0.0
    res = _lp(cv, A_eq=AA.tocsr(), b_eq=lo, bounds=np.column_stack((lb, ub)),
              method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    x = res.x
    return q2.ReplayResult(
        charge=x[oo["c"]:oo["c"] + N], discharge=x[oo["d"]:oo["d"] + N],
        emergency_purchase=x[oo["h"]:oo["h"] + N], curtailment=x[oo["r"]:oo["r"] + N],
        unused_planned_purchase=x[oo["w"]:oo["w"] + N],
        stored_energy=x[oo["e"]:oo["e"] + N + 1],
    )


losses_load: list[np.ndarray] = []
losses_pv: list[np.ndarray] = []
losses_penalty: list[np.ndarray] = []
results = []
initial_energy = q2.E_INITIAL
t_start = time.time()

for day_index, day in enumerate(data.dates):
    hard_terminal = day == q2.YEAR_END
    history_dates = data.dates[:day_index]

    load_scores = q2.historical_scores(
        losses_load, day_index, len(q2.DECAY_DAY_CANDIDATES), data.dates, True
    )
    pv_scores = q2.historical_scores(
        losses_pv, day_index, len(q2.DECAY_DAY_CANDIDATES), data.dates
    )
    penalty_scores = q2.historical_scores(
        losses_penalty, day_index, len(penalties), data.dates
    )
    load_index = q2.choose_candidate(load_scores, default_decay)
    pv_index = q2.choose_candidate(pv_scores, default_decay)
    penalty_index = q2.choose_candidate(penalty_scores, default_penalty)

    load_candidates, pv_candidates = engine.candidates(
        day, history_dates, data.actual_load_energy[:day_index],
        data.actual_photovoltaic_energy[:day_index],
        data.cold_start_load_energy, data.cold_start_photovoltaic_energy,
    )
    if cfg["source"] == "hist":
        fc = q2.Forecast(
            load_candidates[load_index], pv_candidates[pv_index],
            engine.latest_training_date(day, history_dates), False, False,
        )
    elif cfg["source"] == "cold":
        fc = q2.Forecast(
            data.cold_start_load_energy, data.cold_start_photovoltaic_energy,
            None, True, True,
        )
    elif cfg["source"] == "actual":  # perfect information
        fc = q2.Forecast(
            data.actual_load_energy[day_index], data.actual_photovoltaic_energy[day_index],
            data.dates[day_index - 1] if day_index else None, False, False,
        )
    elif cfg["source"] == "perfect_load":
        fl = np.maximum(data.cold_start_load_energy * 0.0 + data.actual_load_energy[day_index], 0.0)
        fc = q2.Forecast(fl, pv_candidates[pv_index],
                         engine.latest_training_date(day, history_dates), False, False)
    else:  # perfect PV forecast
        fc = q2.Forecast(load_candidates[load_index],
                         data.actual_photovoltaic_energy[day_index],
                         engine.latest_training_date(day, history_dates), False, False)

    if hard_terminal:
        plan = q2.solve_validated_plan(day, price, fc, initial_energy, 0.0, True)
        plans = [plan]
    else:
        plans = [
            q2.solve_validated_plan(day, price, fc, initial_energy, float(p), False)
            for p in penalties
        ]
    selected_plan = plans[min(penalty_index, len(plans) - 1)]

    actual_load = data.actual_load_energy[day_index]
    actual_pv = data.actual_photovoltaic_energy[day_index]
    if cfg.get("rt_battery"):
        replays = [rt_dispatch(g, actual_load, actual_pv, initial_energy, hard_terminal)
                   for g in (p.grid_purchase for p in plans)]
    else:
        replays = [
            q2.replay_actual_day(p, actual_load, actual_pv, initial_energy) for p in plans
        ]
    planned_costs = np.array([float(price @ p.grid_purchase) for p in plans])
    emergency_costs = np.array([
        float(q2.EMERGENCY_PRICE_MULTIPLIER * (price @ r.emergency_purchase))
        for r in replays
    ])
    realized = planned_costs + emergency_costs
    result = q2.DailyResult(
        day=day, forecast=fc, plan=selected_plan,
        replay=replays[min(penalty_index, len(replays) - 1)],
        actual_load_energy=actual_load, actual_photovoltaic_energy=actual_pv,
        planned_purchase_cost=float(planned_costs[min(penalty_index, len(plans) - 1)]),
        emergency_purchase_cost=float(emergency_costs[min(penalty_index, len(plans) - 1)]),
        realized_total_cost=float(realized[min(penalty_index, len(plans) - 1)]),
        hyperparameters=q2.Hyperparameters(
            q2.DECAY_DAY_CANDIDATES[load_index], q2.DECAY_DAY_CANDIDATES[pv_index],
            float(penalties[min(penalty_index, len(penalties) - 1)]),
        ),
        parameter_latest_date=None,
    )
    loss_load = np.array([
        q2.pinball_loss(actual_load, x, cfg["q_load"]) for x in load_candidates
    ])
    loss_pv = np.array([
        q2.pinball_loss(actual_pv, x, cfg["q_pv"]) for x in pv_candidates
    ])
    # keep candidate count stable across days for the causal score windows
    if len(losses_load) and len(losses_load[-1]) != len(loss_load):
        raise RuntimeError("candidate count drift")
    losses_load.append(loss_load)
    losses_pv.append(loss_pv)
    losses_penalty.append(realized + terminal_value * np.array([
        abs(r.stored_energy[-1] - q2.E_REFERENCE) for r in replays
    ]))
    results.append(result)
    initial_energy = float(result.replay.stored_energy[-1])

official = results[q2.JANUARY_DAYS:]
plan_cost = sum(r.planned_purchase_cost for r in official)
emerg_cost = sum(r.emergency_purchase_cost for r in official)
total = plan_cost + emerg_cost
em = np.concatenate([r.replay.emergency_purchase for r in official])
gr = np.concatenate([r.plan.grid_purchase for r in official])
un = np.concatenate([r.replay.unused_planned_purchase for r in official])
cur = np.concatenate([r.replay.curtailment for r in official])
p_tile = np.tile(price, len(official))
end_actual = results[-1].replay.stored_energy[-1]
end_plan = results[-1].plan.stored_energy[-1]
soc = np.concatenate([r.replay.stored_energy for r in results])

print(f"variant                 : {VARIANT}")
print(f"334d plan cost          : {plan_cost:.6f} 元")
print(f"334d emergency cost     : {emerg_cost:.6f} 元")
print(f"334d total cost         : {total:.6f} 元")
print(f"emergency energy        : {em.sum():.6f} kWh ({100*em.sum()/gr.sum():.3f}% of purchases)")
print(f"emergency 10-min slots  : {int((em > 1e-7).sum())}")
print(f"wasted planned purchase : {un.sum():.6f} kWh (cost {float(p_tile @ un):.6f} 元)")
print(f"curtailment             : {cur.sum():.6f} kWh")
print(f"year-end actual SOC     : {end_actual:.6f} kWh (plan {end_plan:.6f})")
print(f"Feb-1 initial SOC       : {results[q2.JANUARY_DAYS].replay.stored_energy[0]:.6f} kWh")
print(f"actual SOC range        : {soc.min():.6f} .. {soc.max():.6f}")
print(f"full-year cost (365d)   : {sum(r.realized_total_cost for r in results):.6f} 元")
print(f"wall time               : {time.time() - t_start:.1f} s")

eN = np.array([r.replay.stored_energy[-1] for r in official])
eS = np.array([r.replay.stored_energy[0] for r in official])
print(f"daily end SOC: min={eN.min():.3f} max={eN.max():.3f} mean={eN.mean():.3f} "
      f"| below 5400: {int((eN < 5400 - 1e-6).sum())}/{len(eN)} "
      f"| outside [5400,6600]: {int(((eN < 5400 - 1e-6) | (eN > 6600 + 1e-6)).sum())}")
print(f"daily SOC excursions outside [1200,10800]: "
      f"{int(sum(((r.replay.stored_energy < 1200 - 1e-6) | (r.replay.stored_energy > 10800 + 1e-6)).sum() for r in official))}")

if len(sys.argv) > 2 and sys.argv[2] == "diag":
    from collections import defaultdict
    mon = defaultdict(list)
    for r in results:
        mon[r.day.isoformat()[:7]].append(r)
    print("\nmonth   SOCstart min..max   SOCend mean   emerg kWh      plan kWh     unused kWh  curt kWh")
    for m in sorted(mon):
        v = mon[m]
        print(f"{m} {min(x.replay.stored_energy[0] for x in v):9.1f}..{max(x.replay.stored_energy[0] for x in v):8.1f}"
              f" {np.mean([x.replay.stored_energy[-1] for x in v]):12.1f}"
              f" {sum(x.replay.emergency_purchase.sum() for x in v):12.1f}"
              f" {sum(x.plan.grid_purchase.sum() for x in v):12.1f}"
              f" {sum(x.replay.unused_planned_purchase.sum() for x in v):12.1f}"
              f" {sum(x.replay.curtailment.sum() for x in v):10.1f}")
    d31 = results[-1]
    print("\nDec-31: e0 =", d31.replay.stored_energy[0], " planned eN =", d31.plan.stored_energy[-1],
          " actual eN =", d31.replay.stored_energy[-1])
    print("Dec-31 plan charge/discharge sum =", d31.plan.charge.sum(), d31.plan.discharge.sum())
    print("Dec-31 actual charge/discharge sum =", d31.replay.charge.sum(), d31.replay.discharge.sum())
    print("Dec-31 plan purchase =", d31.plan.grid_purchase.sum(), " emergency =", d31.replay.emergency_purchase.sum())
