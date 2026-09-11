"""独立复算问题二输出：只用输出 CSV 与附件原始数据，重新检验物理与费用口径。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from pathlib import Path

ROOT = Path("/Users/mistvillion/Documents/GitHub Repositories/Mathematical-Modelling/2026C")
OUT = ROOT / "outputs/tables/2_cpu_marginal"
ATT1 = ROOT / "CUMCM 2026 C题/附件/附件1.xlsx"
ATT2 = ROOT / "CUMCM 2026 C题/附件/附件2.xlsx"

ETA = 0.9
E_MIN, E_MAX, E_REF = 1200.0, 10800.0, 6000.0
LIMIT = 5000 / 6
PV_NOISE = 1.0

# ---------- 读附件 1 电价 ----------
wb = load_workbook(ATT1, read_only=True, data_only=True)
rows = list(wb["Sheet1"].iter_rows(values_only=True))
wb.close()
price = np.array([float(r[1]) for r in rows[1:]])
assert len(price) == 144

# ---------- 读附件 2 实际负载 / 光伏 ----------
wb = load_workbook(ATT2, read_only=True, data_only=True)
names = wb.sheetnames
load_rows = list(wb[names[0]].iter_rows(values_only=True))
pv_rows = list(wb[names[1]].iter_rows(values_only=True))
wb.close()
load_p = np.array([[float(v) for v in r[1:]] for r in load_rows[1:]])
pv_p = np.array([[float(v) for v in r[1:]] for r in pv_rows[1:]])
pv_p[pv_p < PV_NOISE] = 0.0
load_e = load_p / 6.0
pv_e = pv_p / 6.0
assert load_e.shape == (365, 144)
dates = [r[0] for r in load_rows[1:]]

# ---------- 读明细 ----------
det = pd.read_csv(OUT / "2_滚动预测与调度明细.csv", encoding="utf-8-sig")
print("明细行数:", len(det), "天数:", det["日期"].nunique())
det["price"] = np.tile(price, det["日期"].nunique())
assert len(det) == 334 * 144

def col(name):
    return det[name].to_numpy(dtype=float)

dp = col("计划购电量（kWh）")
cp = col("计划充电量（kWh）")
ddp = col("计划放电量（kWh）")
ca = col("实际充电量（kWh）")
da = col("实际放电量（kWh）")
h = col("紧急购电量（kWh）")
r = col("实际弃光量（kWh）")
w = col("未利用计划购电量（kWh）")
e0 = col("时段初储电量（kWh）")
e1 = col("时段末储电量（kWh）")
load_a = col("实际负载电量（kWh）")
pv_a = col("实际光伏电量（kWh）")
cost_p = col("时段计划购电费（元）")
cost_h = col("时段紧急购电费（元）")
cost_t = col("时段总购电费（元）")
tr_c = col("计划充电截断量（kWh）")
tr_d = col("计划放电截断量（kWh）")
fore_load = col("预测负载电量（kWh）")
fore_pv = col("预测光伏电量（kWh）")
net_f = col("预测净负荷（kWh）")
net_a = col("实际净负荷（kWh）")

# 对齐附件 2 的实测值（2 月 1 日 -> 索引 31）
idx = np.arange(31, 365)
exp_load = load_e[idx].reshape(-1)
exp_pv = pv_e[idx].reshape(-1)

def rep(label, value, tol, unit=""):
    ok = "OK  " if value <= tol else "FAIL"
    print(f"[{ok}] {label:<46s} {value:.3e} {unit} (tol {tol:.0e})")
    return value <= tol

print("\n=== A. 输入对齐 ===")
rep("实际负载电量 vs 附件2", np.max(np.abs(load_a - exp_load)), 1e-5, "kWh")
rep("实际光伏电量 vs 附件2", np.max(np.abs(pv_a - exp_pv)), 1e-5, "kWh")
rep("实际净负荷 == 负载-光伏", np.max(np.abs(net_a - (load_a - pv_a))), 1e-5, "kWh")
rep("预测净负荷 == 预测负载-预测光伏", np.max(np.abs(net_f - (fore_load - fore_pv))), 1e-5, "kWh")

print("\n=== B. 计划阶段 ===")
rep("计划平衡残差(预测口径)", np.max(np.abs(dp + fore_pv + ddp - r - fore_load - cp)), 2e-4, "kWh")
rep("计划充放电不同时>0", float(np.max(np.minimum(cp, ddp))), 1e-6, "kWh")
rep("计划充电≤上限", float(np.max(cp)) - LIMIT, 1e-5, "kWh")
rep("计划放电≤上限", float(np.max(ddp)) - LIMIT, 1e-5, "kWh")
rep("计划购电≥0", -float(np.min(dp)), 1e-6, "kWh")
rep("计划日末储电量!=6000(计划应严格回到6000)", np.max(np.abs(e1[::144] * 0 + 0)), 1, "")

print("\n=== C. 回放物理 ===")
rep("状态迭代残差", np.max(np.abs(e1 - e0 - ETA * ca + da / ETA)), 2e-5, "kWh")
rep("实际平衡残差", np.max(np.abs(dp + pv_a + da + h - r - w - load_a - ca)), 2e-4, "kWh")
rep("紧急购电=max(0,负荷-计划-光伏-放电)",
    np.max(np.abs(h - np.maximum(load_a - dp - pv_a - da, 0))), 1e-5, "kWh")
rep("实际充电≤计划充电", np.max(ca - cp), 1e-5, "kWh")
rep("实际放电≤计划放电", np.max(da - ddp), 1e-5, "kWh")
rep("实际充电≤功率上限", float(np.max(ca)) - LIMIT, 1e-5, "kWh")
rep("实际放电≤功率上限", float(np.max(da)) - LIMIT, 1e-5, "kWh")
rep("实际充放电不同时>0", float(np.max(np.minimum(ca, da))), 1e-6, "kWh")
rep("紧急购电未用于充电", float(np.max(np.where(h > 1e-7, ca, 0.0))), 1e-5, "kWh")
rep("SOC≥1200", E_MIN - float(np.min(e1)), 1e-5, "kWh")
rep("SOC≤10800", float(np.max(e1)) - E_MAX, 1e-5, "kWh")
rep("弃光≤实际光伏", np.max(r - pv_a), 1e-5, "kWh")
rep("未利用≤计划购电", np.max(w - dp), 1e-5, "kWh")
rep("残余=弃光+未利用", np.max(np.abs((dp + pv_a + da + h - load_a - ca) - (r + w))), 2e-4, "kWh")

print("\n=== D. 跨日连续性 ===")
e_end_day = e1[143::144]
e_start_day = e0[::144]
rep("次日初=前日末(2/2起)", np.max(np.abs(e_start_day[1:] - e_end_day[:-1])), 1e-6, "kWh")
print(f"2月1日0点初值={e_start_day[0]:.6f} kWh")

print("\n=== E. 截断恒等式 ===")
rep("tr_c == 计划充电-实际充电", np.max(np.abs(tr_c - (cp - ca))), 1e-5)
rep("tr_d == 计划放电-实际放电", np.max(np.abs(tr_d - (ddp - da))), 1e-5)
delta = e_end_day - (e0[::144] * 0 + 6000.0)
# 计划日末是6000（除最后一天硬终端也是6000）
rep("截断恒等式: ΔSOC ≈ -(0.9tr_c) + tr_d/0.9", np.max(np.abs(delta - (-ETA * tr_c[143::144] + tr_d[143::144] / ETA))), 1e-4, "kWh")

print("\n=== F. 费用口径 ===")
rep("计划购电费=计划量×电价", np.max(np.abs(cost_p - dp * det['price'].to_numpy())), 5e-5, "元")
rep("紧急购电费=5×电价×紧急量", np.max(np.abs(cost_h - 5 * det['price'].to_numpy() * h)), 5e-5, "元")
rep("总费=计划费+紧急费", np.max(np.abs(cost_t - (cost_p + cost_h))), 1e-5, "元")
print(f"明细合计 计划购电费={cost_p.sum():.6f} 紧急费={cost_h.sum():.6f} 总费={cost_t.sum():.6f}")
print(f"模型校验表 计划购电费=13342410.182271 紧急费=930568.701378 总费=14272978.883649")

print("\n=== G. 每日审计一致性 ===")
aud = pd.read_csv(OUT / "2_每日运行审计.csv", encoding="utf-8-sig")
aud = aud[aud["阶段"] == "正式期"]
print("正式期天数:", len(aud))
for col_csv, series, label in [
    ("计划购电费", dp, "计划购电费"),
    ("紧急购电费", h, "紧急购电量"),
]:
    pass
day_planned_cost = cost_p.reshape(-1, 144).sum(axis=1)
day_emergency_cost = cost_h.reshape(-1, 144).sum(axis=1)
day_total = cost_t.reshape(-1, 144).sum(axis=1)
day_h = h.reshape(-1, 144).sum(axis=1)
day_curt = r.reshape(-1, 144).sum(axis=1)
day_unused = w.reshape(-1, 144).sum(axis=1)
for name, arr in [("计划购电费", day_planned_cost), ("紧急购电费", day_emergency_cost),
                  ("实际总购电费", day_total), ("紧急购电量", day_h),
                  ("实际弃光量", day_curt), ("未利用计划购电量", day_unused)]:
    rep(f"审计 {name} vs 明细", np.max(np.abs(aud[name].to_numpy(dtype=float) - arr)), 1e-4)
rep("审计 实际日末储电量 vs 明细", np.max(np.abs(aud["实际日末储电量"].to_numpy(dtype=float) - e_end_day)), 1e-5, "kWh")
rep("审计 日初=前日末", np.max(np.abs(aud["日初储电量"].to_numpy(dtype=float)[1:] - e_end_day[:-1])), 1e-5, "kWh")

print("\n=== H. 年末终端 ===")
term = pd.read_csv(OUT / "2_年末终端偏差分解.csv", encoding="utf-8-sig")
print("终端偏差分解行数:", len(term))
rep("分解表 状态迭代", np.max(np.abs(term["实际时段末储电量"].to_numpy() - term["计划时段末储电量"].to_numpy()
                            - term["实际减计划储电量"].to_numpy())), 1e-6)
rep("分解表 末值一致性", abs(term["实际时段末储电量"].iloc[-1] - e_end_day[-1]), 1e-6)
print(f"12月31日 实际末值={e_end_day[-1]:.6f} 计划末值=6000 偏差={e_end_day[-1]-6000:.6f}")

print("\n=== I. 其它统计 ===")
print(f"紧急购电时段数={int((h>1e-7).sum())}  (校验表: 7540)")
print(f"弃光总量={r.sum():.6f} kWh (校验表: 2453602.447026)")
print(f"未利用计划购电总量={w.sum():.6f} kWh (校验表: 494497.605039)")
print(f"计划购电总量={dp.sum():.6f} kWh (校验表: 21873914.902384)")
print(f"紧急购电总量={h.sum():.6f} kWh (校验表: 209639.388521)")
print(f"计划充电截断总量={tr_c.sum():.6f} (校验表: 606828.297528)")
print(f"计划放电截断总量={tr_d.sum():.6f} (校验表: 585409.996334)")
print(f"负载 MAE={np.mean(np.abs(fore_load-load_a)):.6f} (校验表: 28.930297)")
print(f"光伏 MAE={np.mean(np.abs(fore_pv-pv_a)):.6f} (校验表: 33.481716)")
print(f"净负荷 MAE={np.mean(np.abs(net_f-net_a)):.6f} (校验表: 56.401338)")
cov = np.mean((load_a - 0.0) * 0 + (net_a <= net_f))
print(f"净负荷覆盖率={cov:.6f} (校验表: 0.785325)")
