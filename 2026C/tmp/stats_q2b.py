"""重建计划 SOC 轨迹；检查计划/实际边界命中与预测分位数覆盖。"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("/Users/mistvillion/Documents/GitHub Repositories/Mathematical-Modelling/2026C")
OUT = ROOT / "outputs/tables/2_cpu_marginal"
det = pd.read_csv(OUT / "2_滚动预测与调度明细.csv", encoding="utf-8-sig")
num = ["预测负载电量（kWh）", "预测光伏电量（kWh）", "实际负载电量（kWh）", "实际光伏电量（kWh）",
       "计划购电量（kWh）", "计划充电量（kWh）", "计划放电量（kWh）", "实际充电量（kWh）",
       "实际放电量（kWh）", "紧急购电量（kWh）", "实际弃光量（kWh）", "未利用计划购电量（kWh）",
       "时段初储电量（kWh）", "时段末储电量（kWh）"]
g = {c: det[c].to_numpy(float) for c in num}
days = det["日期"].nunique()
ETA = 0.9
E0 = g["时段初储电量（kWh）"].reshape(days, 144)
E1 = g["时段末储电量（kWh）"].reshape(days, 144)
CP = g["计划充电量（kWh）"].reshape(days, 144)
DDP = g["计划放电量（kWh）"].reshape(days, 144)

# 计划轨迹 = 日初(实际) + 累加(0.9*计划充电 - 计划放电/0.9)
plan_soc = np.empty((days, 145))
plan_soc[:, 0] = E0[:, 0]
plan_soc[:, 1:] = E0[:, 0:1] + np.cumsum(ETA * CP - DDP / ETA, axis=1)
act_soc = np.concatenate([E0, E1[:, -1:]], axis=1)

print("=== 计划 SOC 轨迹（由计划充放电重建）===")
print(f"计划 SOC min={plan_soc.min():.6f} max={plan_soc.max():.6f}")
print(f"计划 SOC 低于1200 的时段数: {int((plan_soc<1199.999).sum())}, 高于10800: {int((plan_soc>10800.001).sum())}")
print(f"计划日末 SOC 唯一值: {np.unique(np.round(plan_soc[:,-1],6))[:5]} (应为 6000)")
print(f"计划 SOC == 10800 的时段数: {int((np.abs(plan_soc-10800)<1e-4).sum())}")
print(f"计划 SOC == 1200  的时段数: {int((np.abs(plan_soc-1200)<1e-4).sum())}")
print(f"实际 SOC == 10800 的时段数: {int((np.abs(act_soc-10800)<1e-4).sum())}")
print(f"实际 SOC == 1200  的时段数: {int((np.abs(act_soc-1200)<1e-4).sum())}")
diff = act_soc - plan_soc
print(f"实际-计划 SOC 差值: min={diff.min():.3f} max={diff.max():.3f} mean={diff.mean():.3f}")
print(f"|差值|>100 kWh 的时段数: {int((np.abs(diff)>100).sum())}")

print("\n=== 计划弃光隐含值的越界检查 ===")
curt = g["计划购电量（kWh）"] + g["预测光伏电量（kWh）"] + g["计划放电量（kWh）"] - g["预测负载电量（kWh）"] - g["计划充电量（kWh）"]
excess = curt - g["预测光伏电量（kWh）"]
neg = -curt
print(f"max(计划弃光-预测光伏)={excess.max():.3e}  对应时段数(>1e-4)={int((excess>1e-4).sum())}")
print(f"max(-计划弃光)={neg.max():.3e}  对应时段数(>1e-4)={int((neg>1e-4).sum())}")
if (excess>1e-4).any():
    idx = np.flatnonzero(excess>1e-4)[:5]
    for i in idx:
        print(f"   {det['日期'].iloc[i]} {det['时间段'].iloc[i]} 弃光={curt[i]:.6f} 预测光伏={g['预测光伏电量（kWh）'][i]:.6f} 超出={excess[i]:.6f}")

print("\n=== 光伏/负载分位数覆盖（分白天/夜间）===")
day_mask = g["预测光伏电量（kWh）"] > 0
print(f"全时段: 光伏覆盖率={np.mean(g['实际光伏电量（kWh）']<=g['预测光伏电量（kWh）']):.4f}, 白天时段数={int(day_mask.sum())}/{len(day_mask)}")
print(f"白天: 光伏覆盖率={np.mean(g['实际光伏电量（kWh）'][day_mask]<=g['预测光伏电量（kWh）'][day_mask]):.4f}")
print(f"全时段: 负载覆盖率={np.mean(g['实际负载电量（kWh）']<=g['预测负载电量（kWh）']):.4f}")
print(f"全时段: 净负荷覆盖率={np.mean(g['实际负载电量（kWh）']-g['实际光伏电量（kWh）']<=g['预测负载电量（kWh）']-g['预测光伏电量（kWh）']):.4f}")

print("\n=== 12月31日截断诊断核对 ===")
term = pd.read_csv(OUT / "2_年末终端偏差分解.csv", encoding="utf-8-sig")
print(f"充电截断合计={term['充电截断量'].sum():.6f} 放电截断合计={term['放电截断量'].sum():.6f}")
print(f"0.9*(-充电截断)+ 放电截断/0.9 = {-0.9*term['充电截断量'].sum() + term['放电截断量'].sum()/0.9:.6f} (应为 212.852394)")

print("\n=== 日末实际-计划储电量偏差的分布 ===")
aud = pd.read_csv(OUT / "2_每日运行审计.csv", encoding="utf-8-sig")
formal = aud[aud["阶段"] == "正式期"]
delta = formal["实际减计划日末储电量"].to_numpy(float)
print(f"偏差: min={delta.min():.3f} max={delta.max():.3f} mean={delta.mean():.3f} 年末={delta[-1]:.3f}")
print(f"偏差>0 的天数={int((delta>0).sum())}, <0 的天数={int((delta<0).sum())}")
