"""Independent reproduction of the Q2 forecasts, plans and replays.

For a sample of days this recomputes, from the raw attachments only:
  * the causal weighted-quantile forecast for every decay candidate,
  * the daily MILP plan for each terminal-penalty candidate,
  * the replay of the actual day,
and compares against the published detail CSV.
"""

from __future__ import annotations

import csv
from datetime import date

import numpy as np
import openpyxl
from scipy.optimize import Bounds, LinearConstraint, linprog
from scipy.sparse import lil_matrix

DT = 1 / 6
E_MIN, E_MAX, E_REF = 1200.0, 10800.0, 6000.0
ETA_C = ETA_D = 0.9
LIM = 5000 / 6
N = 144
PEN = 5.0
PV_NOISE = 1.0
DECAYS = (3.0, 5.0, 7.0, 10.0, 14.0, 21.0, 28.0, 42.0, 56.0, 84.0)
PENALTIES_MULT = (0.5, 1.0, 2.0, 5.0)


def read_actual(sheet):
    wb = openpyxl.load_workbook(
        "CUMCM 2026 C题/附件/附件2.xlsx", read_only=True, data_only=True
    )
    rows = list(wb[sheet].iter_rows(values_only=True))
    wb.close()
    dates = [r[0].date() for r in rows[1:]]
    return dates, np.array([[float(v) for v in r[1:]] for r in rows[1:]])


dates, load_p = read_actual("小区负载")
_, pv_p = read_actual("光伏发电实际功率")
pv_p[pv_p < PV_NOISE] = 0.0
load = load_p * DT
pv = pv_p * DT
wb = openpyxl.load_workbook(
    "CUMCM 2026 C题/附件/附件1.xlsx", read_only=True, data_only=True
)
rows = list(wb["Sheet1"].iter_rows(values_only=True))[1:]
wb.close()
price = np.array([float(r[1]) for r in rows])
cold_load = np.array([float(r[2]) for r in rows]) * DT
cold_pv = np.array([float(r[3]) for r in rows]) * DT
cold_pv[cold_pv < PV_NOISE * DT] = 0.0
pmax = price.max()
penalties = pmax * np.asarray(PENALTIES_MULT)

detail = list(csv.DictReader(open("outputs/tables/2_滚动预测与调度明细.csv", encoding="utf-8-sig")))


def is_high(d):
    return d.weekday() <= 3 or d.weekday() == 6


def wquantile(values, ages, q):
    """weighted quantile, replicating the project convention (lower cdf crossing)"""
    w = np.exp(-ages / np.asarray(DECAYS)[:, None])
    order = np.argsort(values, axis=0, kind="stable")
    cum = np.cumsum(w[:, order], axis=1)
    idx = np.argmax(cum >= q * cum[:, -1:, :], axis=1)
    return order[idx, np.arange(values.shape[1])[None, :]]


def forecast_candidates(target_idx):
    """all decay candidates for the load (q=0.8) and PV (q=0.2) forecast."""
    target = dates[target_idx]
    hist = np.arange(target_idx)
    if len(hist) == 0:
        return (np.repeat(cold_load[None, :], len(DECAYS), axis=0),
                np.repeat(cold_pv[None, :], len(DECAYS), axis=0))
    ages = np.array([(target - dates[i]).days for i in hist], dtype=float)
    same = np.array([is_high(dates[i]) == is_high(target) for i in hist])
    if same.any():
        lv, la = load[hist[same]], ages[same]
    else:
        lv, la = cold_load[None, :], np.ones(1)
    lq = lv[wquantile(lv, la, 0.8), np.arange(N)[None, :]]
    pv_v, pv_a = pv[hist], ages
    vq = pv_v[wquantile(pv_v, pv_a, 0.2), np.arange(N)[None, :]]
    return lq, vq


def daily_plan(fl, fv, e0, lam):
    """full daily planned model: purchase cost + lambda_E * |e_N - 6000|."""
    nvar = 6 * N + 3
    o = dict(g=0, c=N, d=2 * N, r=3 * N, u=4 * N, e=5 * N)
    zp, zm = 6 * N + 1, 6 * N + 2
    cvec = np.zeros(nvar)
    cvec[o["g"]:o["g"] + N] = price
    cvec[zp] = cvec[zm] = lam
    lb = np.zeros(nvar)
    ub = np.full(nvar, np.inf)
    ub[o["c"]:o["c"] + N] = LIM
    ub[o["d"]:o["d"] + N] = LIM
    ub[o["r"]:o["r"] + N] = fv
    ub[o["u"]:o["u"] + N] = 1.0
    lb[o["e"]:o["e"] + N + 1] = E_MIN
    ub[o["e"]:o["e"] + N + 1] = E_MAX
    lb[o["e"]] = ub[o["e"]] = e0
    A = lil_matrix((2 * N + 1, nvar))
    lo = np.full(2 * N + 1, -np.inf)
    hi = np.full(2 * N + 1, np.inf)
    for t in range(N):
        A[t, o["g"] + t] = 1
        A[t, o["c"] + t] = -1
        A[t, o["d"] + t] = 1
        A[t, o["r"] + t] = -1
        lo[t] = hi[t] = fl[t] - fv[t]
        row = N + t
        A[row, o["c"] + t] = -ETA_C
        A[row, o["d"] + t] = 1 / ETA_D
        A[row, o["e"] + t] = -1
        A[row, o["e"] + t + 1] = 1
        lo[row] = hi[row] = 0.0
    A[2 * N, o["e"] + N] = 1.0
    A[2 * N, zp] = -1.0
    A[2 * N, zm] = 1.0
    lo[2 * N] = hi[2 * N] = E_REF
    res = linprog(cvec, A_eq=A.tocsr(), b_eq=lo, bounds=np.column_stack((lb, ub)),
                  method="highs")
    assert res.success, res.message
    x = res.x
    return (x[o["g"]:o["g"] + N], x[o["c"]:o["c"] + N], x[o["d"]:o["d"] + N],
            x[o["r"]:o["r"] + N], x[o["e"]:o["e"] + N + 1], float(res.fun))


def replay(g, cp, dp, la, va, e0):
    c = np.zeros(N); d = np.zeros(N); h = np.zeros(N); e = np.empty(N + 1)
    e[0] = e0
    for t in range(N):
        cur = e[t]
        deficit = max(la[t] - g[t] - va[t], 0.0)
        d[t] = min(dp[t], LIM, max(ETA_D * (cur - E_MIN), 0.0), deficit)
        h[t] = max(la[t] - g[t] - va[t] - d[t], 0.0)
        surplus = max(g[t] + va[t] + d[t] - la[t], 0.0)
        c[t] = min(cp[t], LIM, max((E_MAX - cur) / ETA_C, 0.0), surplus)
        e[t + 1] = cur + ETA_C * c[t] - d[t] / ETA_D
    return c, d, h, e


# published per-day lookups
by_day = {}
for k in range(334):
    by_day[detail[k * N]["日期"]] = k

sample = ["2025-02-01", "2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21", "2025-07-04", "2025-10-31"]
print(f"{'day':12s} {'fcL ok':>7s} {'fcV ok':>7s} {'plan ok':>8s} {'replay ok':>9s} "
      f"{'rhoL':>6s} {'rhoV':>6s} {'pen':>8s} {'maxdiff':>10s}")
for day in sample:
    k = by_day[day]
    blk = detail[k * N:(k + 1) * N]
    ti = dates.index(date.fromisoformat(day))
    pub_fl = np.array([float(r["预测负载电量（kWh）"]) for r in blk])
    pub_fv = np.array([float(r["预测光伏电量（kWh）"]) for r in blk])
    pub_g = np.array([float(r["计划购电量（kWh）"]) for r in blk])
    pub_cp = np.array([float(r["计划充电量（kWh）"]) for r in blk])
    pub_dp = np.array([float(r["计划放电量（kWh）"]) for r in blk])
    pub_ca = np.array([float(r["实际充电量（kWh）"]) for r in blk])
    pub_da = np.array([float(r["实际放电量（kWh）"]) for r in blk])
    pub_h = np.array([float(r["紧急购电量（kWh）"]) for r in blk])
    pub_e = np.array([float(r["时段初储电量（kWh）"]) for r in blk] + [float(blk[-1]["时段末储电量（kWh）"])])
    # recover chosen decays and penalty from the audit csv
    audit = {r["日期"]: r for r in csv.DictReader(open("outputs/tables/2_每日运行审计.csv", encoding="utf-8-sig"))}
    rhoL = float(audit[day]["负载衰减天数"])
    rhoV = float(audit[day]["光伏衰减天数"])
    lam = float(audit[day]["终端惩罚"])

    lq, vq = forecast_candidates(ti)
    li = DECAYS.index(rhoL)
    vi = DECAYS.index(rhoV)
    d_fl = float(np.abs(lq[li] - pub_fl).max())
    d_fv = float(np.abs(vq[vi] - pub_fv).max())

    g, cp, dp, rp, ep, obj = daily_plan(pub_fl, pub_fv, pub_e[0], lam)
    dg = float(np.abs(g - pub_g).max())
    dc = float(np.abs(cp - pub_cp).max())
    dd = float(np.abs(dp - pub_dp).max())
    de = float(np.abs(ep - pub_e).max())
    d_plan = max(dg, dc, dd, de)
    ca, da, h, e = replay(g, cp, dp, load[ti], pv[ti], pub_e[0])
    d_rep = max(float(np.abs(ca - pub_ca).max()), float(np.abs(da - pub_da).max()),
                float(np.abs(h - pub_h).max()), float(np.abs(e - pub_e).max()))
    print(f"{day:12s} {d_fl:7.2e} {d_fv:7.2e} {d_plan:8.2e} {d_rep:9.2e} "
          f"{rhoL:6.0f} {rhoV:6.0f} {lam:8.4f} {max(d_fl, d_fv, d_plan, d_rep):10.2e}"
          f"   [g {dg:.1e} c {dc:.1e} d {dd:.1e} e {de:.1e}]")
