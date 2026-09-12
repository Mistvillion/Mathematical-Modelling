"""Independent audit of the Q2 deliverables (no reuse of project code).

Rebuilds the actual-data tensors from 附件2, then checks the frozen plan /
replay / cost bookkeeping written out by scripts_tables/2_rolling_optimization.py.
"""

from __future__ import annotations

import csv
from datetime import date, datetime

import numpy as np
import openpyxl
from openpyxl.utils.datetime import from_excel

DT = 1 / 6
E_MIN, E_MAX, E0 = 1200.0, 10800.0, 6000.0
ETA_C = ETA_D = 0.9
LIM = 5000 / 6
N = 144
PEN = 5.0

# ---------- raw actual data ----------
def read_actual(sheet):
    wb = openpyxl.load_workbook(
        "CUMCM 2026 C题/附件/附件2.xlsx", read_only=True, data_only=True
    )
    rows = list(wb[sheet].iter_rows(values_only=True))
    wb.close()
    dates = [r[0].date() if isinstance(r[0], datetime) else r[0] for r in rows[1:]]
    vals = np.array([[float(v) for v in r[1:]] for r in rows[1:]])
    return dates, vals * DT


dates, load = read_actual("小区负载")
_, pv = read_actual("光伏发电实际功率")
pv[pv < 1.0 * DT] = 0.0
print("actual days:", len(dates), dates[0], dates[-1])
print("annual actual load  =", load.sum(), "kWh")
print("annual actual PV    =", pv.sum(), "kWh")
print("annual net load     =", (load - pv).sum(), "kWh")

wb = openpyxl.load_workbook(
    "CUMCM 2026 C题/附件/附件1.xlsx", read_only=True, data_only=True
)
rows = list(wb["Sheet1"].iter_rows(values_only=True))[1:]
wb.close()
price = np.array([float(r[1]) for r in rows])
cold_load = np.array([float(r[2]) for r in rows]) * DT
cold_pv = np.array([float(r[3]) for r in rows]) * DT
cold_pv[cold_pv < 1.0 * DT] = 0.0

# ---------- detail csv ----------
with open("outputs/tables/2_滚动预测与调度明细.csv", encoding="utf-8-sig") as fh:
    det = list(csv.DictReader(fh))
print("\ndetail rows:", len(det), "expected", 334 * 144)
assert len(det) == 334 * 144

def col(name):
    return np.array([float(r[name]) for r in det])

day_list = [r["日期"] for r in det]
g = col("计划购电量（kWh）")
c = col("实际充电量（kWh）")
d = col("实际放电量（kWh）")
h = col("紧急购电量（kWh）")
r_ = col("实际弃光量（kWh）")
w = col("未利用计划购电量（kWh）")
e0 = col("时段初储电量（kWh）")
e1 = col("时段末储电量（kWh）")
la = col("实际负载电量（kWh）")
va = col("实际光伏电量（kWh）")
lp = col("预测负载电量（kWh）")
vp = col("预测光伏电量（kWh）")
cp = col("计划充电量（kWh）")
dp = col("计划放电量（kWh）")
pc = col("时段计划购电费（元）")
ec = col("时段紧急购电费（元）")

# tiled price
p_tile = np.tile(price, 334)
# tiled actuals in date order
idx = {d_: i for i, d_ in enumerate(dates)}
day_seq = [day_list[k * N] for k in range(334)]
day_idx = np.array([idx[date.fromisoformat(s)] for s in day_seq])
la_ref = np.concatenate([load[i] for i in day_idx])
va_ref = np.concatenate([pv[i] for i in day_idx])
print("actual load matches 附件2:", np.allclose(la, la_ref, atol=5e-7))
print("actual PV   matches 附件2:", np.allclose(va, va_ref, atol=5e-7))
print("price matches 附件1:", np.allclose(p_tile, np.tile(price, 334), atol=1e-12))

print("\n--- cost bookkeeping ---")
print("sum p*g          =", float(np.sum(p_tile * g)))
print("sum 5p*h         =", float(np.sum(PEN * p_tile * h)))
print("total            =", float(np.sum(p_tile * g) + np.sum(PEN * p_tile * h)))
print("csv 时段计划购电费 sum =", float(pc.sum()), "| 时段紧急购电费 sum =", float(ec.sum()))
print("planned+emerg csv      =", float(pc.sum() + ec.sum()))

print("\n--- replay identities ---")
h_exp = np.maximum(la - g - va - d, 0.0)
print("max |h - max(0, load-g-pv-d)| =", float(np.max(np.abs(h - h_exp))))
bal = g + va + d + h - r_ - w - la - c
print("max |actual balance residual| =", float(np.max(np.abs(bal))))
st = e1 - e0 - ETA_C * c + d / ETA_D
print("max |state transition residual| =", float(np.max(np.abs(st))))
print("max (c - cp)  =", float(np.max(c - cp)), "(must be <=0)")
print("max (d - dp)  =", float(np.max(d - dp)), "(must be <=0)")
print("max c =", float(c.max()), "limit", LIM, "| max d =", float(d.max()))
print("simultaneous ch/dis =", float(np.minimum(c, d).max()))
print("max (h>0 & c>0) violations =", int(np.sum((h > 1e-7) & (c > 1e-6))))
print("SOC min/max =", e0.min(), e1.max())
print("max jump |e0_next - e1_prev| =",
      float(np.abs(np.concatenate([e0[1:] - e1[:-1], e0[N:] - e1[N - 1:-1]])).max()))

print("\n--- decoupled cross-day check (day blocks) ---")
cross = []
for k in range(1, 334):
    cross.append(abs(e0[k * N] - e1[k * N - 1]))
print("max cross-day SOC jump =", max(cross))

print("\n--- curtailment / unused purchase bounds ---")
print("max (r - pv_actual) =", float(np.max(r_ - va)))
print("max (w - g) =", float(np.max(w - g)))
print("sum r =", float(r_.sum()), "| sum w =", float(w.sum()))

print("\n--- non-anticipativity in detail csv ---")
tr = [r["训练样本最晚日期"] for r in det]
pa = [r["参数验证最晚日期"] for r in det]
bad_tr = sum(1 for s, d_ in zip(tr, day_list) if s != "附件1冷启动" and s >= d_)
bad_pa = sum(1 for s, d_ in zip(pa, day_list) if s != "预设参数" and s >= d_)
print("rows with training cutoff >= day:", bad_tr)
print("rows with param cutoff >= day:", bad_pa)
print("distinct training cutoffs per day ok:",
      all(len({s for s, d_ in zip(tr, day_list) if d_ == dd}) == 1 for dd in set(day_list)))

print("\n--- daily aggregation cross-check vs 指定日期 summary ---")
by_day = {}
for i, s in enumerate(day_list):
    by_day.setdefault(s, []).append(i)
print("days present:", len(by_day))
for target in ("2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"):
    ii = by_day[target]
    print(f"  {target}: plan total = {g[ii].sum():.6f}, emergency = {h[ii].sum():.6f}, "
          f"e0 = {e0[ii[0]]:.6f}, e24 = {e1[ii[-1]]:.6f}, "
          f"cost = {(p_tile[ii]*g[ii]).sum() + PEN*(p_tile[ii]*h[ii]).sum():.6f}")
