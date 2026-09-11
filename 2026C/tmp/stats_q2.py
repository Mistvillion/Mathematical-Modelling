"""补充统计：边界命中、计划弃光隐含值、紧急购电分布、年初年末衔接。"""
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from pathlib import Path

ROOT = Path("/Users/mistvillion/Documents/GitHub Repositories/Mathematical-Modelling/2026C")
OUT = ROOT / "outputs/tables/2_cpu_marginal"
wb = load_workbook(ROOT / "CUMCM 2026 C题/附件/附件1.xlsx", read_only=True, data_only=True)
price = np.array([float(r[1]) for r in list(wb["Sheet1"].iter_rows(values_only=True))[1:]])
wb.close()
print(f"电价: min={price.min():.4f} max={price.max():.4f} mean={price.mean():.4f} 时段数={len(price)}")

det = pd.read_csv(OUT / "2_滚动预测与调度明细.csv", encoding="utf-8-sig")
g = {c: det[c].to_numpy(float) for c in det.columns if c not in ("日期", "时间段", "负载预测来源", "光伏预测来源", "预测方法", "训练样本最晚日期", "参数验证最晚日期")}
days = det["日期"].nunique()
p = np.tile(price, days)

fore_load, fore_pv = g["预测负载电量（kWh）"], g["预测光伏电量（kWh）"]
dp, cp, ddp = g["计划购电量（kWh）"], g["计划充电量（kWh）"], g["计划放电量（kWh）"]
ca, da, h = g["实际充电量（kWh）"], g["实际放电量（kWh）"], g["紧急购电量（kWh）"]
r, w = g["实际弃光量（kWh）"], g["未利用计划购电量（kWh）"]
e0, e1 = g["时段初储电量（kWh）"], g["时段末储电量（kWh）"]
load_a, pv_a = g["实际负载电量（kWh）"], g["实际光伏电量（kWh）"]

print("\n--- 计划弃光量（由计划平衡隐含）---")
curt_plan = dp + fore_pv + ddp - fore_load - cp
print(f"隐含计划弃光: min={curt_plan.min():.6e} max={curt_plan.max():.6f} 总量={curt_plan.sum():.3f}")
print(f"是否恒在[0,预测光伏]内: {curt_plan.min() >= -1e-6 and np.max(curt_plan - fore_pv) <= 1e-6}")
print(f"计划弃光总量 vs 实际弃光总量: {curt_plan.sum():.1f} vs {r.sum():.1f}")

print("\n--- 储电量边界命中 ---")
tol = 1e-6
print(f"实际SOC=1200 的时段数: {int((np.abs(e1-1200)<1e-4).sum())}, =10800: {int((np.abs(e1-10800)<1e-4).sum())}")
print(f"实际SOC min={e1.min():.6f} max={e1.max():.6f}")
print(f"实际SOC 低于1500 的时段数: {int((e1<1500).sum())}, 高于10500: {int((e1>10500).sum())}")

print("\n--- 紧急购电分布 ---")
act = h > 1e-7
runs = 0
intervals = []
start = None
for i, a in enumerate(act):
    if a and start is None:
        start = i
    elif not a and start is not None:
        runs += 1; intervals.append((start, i)); start = None
if start is not None:
    runs += 1; intervals.append((start, len(act)))
# 按天切分统计（与脚本口径一致：不跨日合并）
runs_day = 0
for d in range(days):
    a = act[d*144:(d+1)*144]
    if a.any():
        runs_day += int(np.sum(a[1:] & ~a[:-1])) + int(a[0])
print(f"紧急购电时段数={int(act.sum())} 连续区间数(不跨日)={runs_day} 连续区间数(跨日合并)={runs}")
lengths = [b-a for a, b in intervals]
print(f"区间长度分布: mean={np.mean(lengths):.2f} median={np.median(lengths)} max={max(lengths)} 长度1的区间数={sum(1 for L in lengths if L==1)}")
print(f"紧急购电总量={h.sum():.3f} kWh 费用={5*(p*h).sum():.3f} 元")
day_h = h.reshape(days,144).sum(axis=1)
top = np.argsort(-day_h)[:8]
print("紧急购电最多的日期:")
for i in top:
    print(f"  {det['日期'].iloc[i*144]} 紧急购电={day_h[i]:10.3f} kWh 实际负载={load_a.reshape(days,144).sum(axis=1)[i]:10.1f} 实际光伏={pv_a.reshape(days,144).sum(axis=1)[i]:10.1f}")
print(f"紧急购电为0的天数={int((day_h<1e-7).sum())}")

print("\n--- 同时发生检查 ---")
print(f"同日时段 紧急购电>0 且 未利用计划购电>0: {int(((h>1e-7)&(w>1e-7)).sum())}")
print(f"同日时段 紧急购电>0 且 弃光>0: {int(((h>1e-7)&(r>1e-7)).sum())}")
print(f"同日时段 未利用>0 且 弃光>0: {int(((w>1e-7)&(r>1e-7)).sum())}")

print("\n--- 全年能量总量（2/1-12/31）---")
print(f"实际负载 {load_a.sum():15.1f} kWh")
print(f"实际光伏 {pv_a.sum():15.1f} kWh")
print(f"计划购电 {dp.sum():15.1f} kWh")
print(f"紧急购电 {h.sum():15.1f} kWh")
print(f"实际充电 {ca.sum():15.1f} kWh  实际放电 {da.sum():15.1f} kWh")
print(f"弃光     {r.sum():15.1f} kWh  未利用计划购电 {w.sum():15.1f} kWh")
print(f"计划充电 {cp.sum():15.1f} kWh  计划放电 {ddp.sum():15.1f} kWh  计划弃光 {curt_plan.sum():15.1f} kWh")
print(f"光伏弃光率(实际)={r.sum()/pv_a.sum():.4f}  计划弃光率={curt_plan.sum()/fore_pv.sum():.4f}")

print("\n--- 审计表：计划日末储电量、年初年末 ---")
aud = pd.read_csv(OUT / "2_每日运行审计.csv", encoding="utf-8-sig")
print("计划日末储电量 唯一值:", np.unique(np.round(aud["计划日末储电量"].to_numpy(float), 6))[:10])
jan31 = aud[aud["日期"] == "2025-01-31"].iloc[0]
feb1 = aud[aud["日期"] == "2025-02-01"].iloc[0]
print(f"1月31日 实际日末={jan31['实际日末储电量']:.6f}  计划日末={jan31['计划日末储电量']:.6f}")
print(f"2月1日 日初={feb1['日初储电量']:.6f}  -> 衔接{'正确' if abs(jan31['实际日末储电量']-feb1['日初储电量'])<1e-6 else '错误'}")
print(f"1月31日阶段={jan31['阶段']} 2月1日阶段={feb1['阶段']}")
e_end = e1[143::144]
print(f"2月1日初={e_end[0]-0:.6f}? 实际={e0[0]:.6f}")
print(f"年内实际SOC日末: min={e_end.min():.3f} max={e_end.max():.3f} mean={e_end.mean():.3f}")
print(f"年末(12/31)={e_end[-1]:.6f}")

print("\n--- 分位数覆盖 ---")
net_a = load_a - pv_a
print(f"净负荷覆盖率 = {np.mean(net_a <= fore_load-fore_pv):.6f}")
print(f"负载覆盖率(实际<=预测) = {np.mean(load_a <= fore_load):.6f}")
print(f"光伏覆盖率(实际<=预测) = {np.mean(pv_a <= fore_pv):.6f}")
