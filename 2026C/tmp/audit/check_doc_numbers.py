"""Verify every number written into docs/Q2全知视角.md 7.12 against the CSVs."""
import csv, re, numpy as np
doc = open("docs/Q2全知视角.md", encoding="utf-8").read()
daily = list(csv.DictReader(open("outputs/tables/2_全知视角逐日费用.csv", encoding="utf-8-sig")))
chk = {r["校验项"]: r["数值"] for r in csv.DictReader(open("outputs/tables/2_全知视角模型校验.csv", encoding="utf-8-sig"))}
cmp_ = {r["指标"]: r for r in csv.DictReader(open("outputs/tables/2_全知视角与因果模型对比.csv", encoding="utf-8-sig"))}
spec = list(csv.DictReader(open("outputs/tables/2_全知视角指定日期购电量及全天结果.csv", encoding="utf-8-sig")))

def has(s):
    ok = s in doc
    print(("  OK   " if ok else "  MISS ") + s)
    return ok

print("== 7.12.1 headline ==")
for s in ["12 241 228.517471", "20 210 421.636692", "6 745 087.078405", "5 464 060.533508",
          "1 281 626.544897", "990 168.090312", "36 650.384783", "0.330772", "0.646678", "2.517"]:
    has(s)

print("== 7.12.2 daily distribution ==")
eN = np.array([float(r["24:00储电量（kWh）"]) for r in daily])
cur = np.array([float(r["弃光量（kWh）"]) for r in daily])
c = np.array([float(r["购电费（元）"]) for r in daily])
d = np.array([r["日期"] for r in daily])
print(f"  real: eN==5400 -> {int((np.abs(eN-5400)<1e-6).sum())}, eN==6600 -> {int((np.abs(eN-6600)<1e-6).sum())}, "
      f"inside -> {int(((eN>5400+1e-6)&(eN<6600-1e-6)).sum())}, eN range {eN.min()}..{eN.max()}")
print(f"  real: days curt>0 -> {int((cur>1e-6).sum())}, December curt -> {cur[[x.startswith('2025-12') for x in d]].sum()}")
print(f"  real: daily cost min {c.min():.1f} ({d[c.argmin()]}), max {c.max():.1f} ({d[c.argmax()]}), mean {c.mean():.1f}")
for s in ["67 天恰好取下界 5400 kWh", "260 天恰好取上界 6600 kWh", "只有 7 天严格落在区间内部",
          "12 148.9 元（2025-04-25）", "59 968.9 元（2025-12-16）", "36 650.4 元/天", "97 天"]:
    has(s)

print("== 7.12.2 monthly table ==")
mm = {}
for r in daily:
    v = mm.setdefault(r["日期"][:7], [0.0, 0.0, 0.0, 0.0])
    v[0] += float(r["计划购电量（kWh）"]); v[1] += float(r["购电费（元）"])
    v[2] += float(r["弃光量（kWh）"]); v[3] += float(r["充电量（kWh）"])
for m in sorted(mm):
    p, cost, cu, chg = mm[m]
    row = f"| {m} | {p:,.1f} | {cost:,.1f} | {cu:,.1f} | {chg:,.1f} |".replace(",", " ")
    print(("  OK   " if row in doc else "  DIFF ") + row)

print("== 7.12.3 comparison table ==")
pairs = [("计划购电费", "12 241 228.517471", "13 342 127.615605", "1 100 899.098134", "8.25%"),
         ("紧急购电费", "0.000000", "930 568.701378", "930 568.701378", None),
         ("334 天总费用", "12 241 228.517471", "14 272 696.316983", "2 031 467.799512", "14.23%"),
         ("计划购电总量", "20 210 421.636692", "21 873 248.235717", "1 662 826.599025", "7.60%"),
         ("弃光总量", "990 168.090312", "2 453 602.447026", "1 463 434.356714", "59.64%"),
         ("紧急购电 10 分钟时段数", "0", "7 540", "7 540", None)]
for row in pairs:
    label, o, cau, diff, rel = row
    src = cmp_[label]
    m = (abs(float(src["全知视角（完美预见）"]) - float(o.replace(" ", ""))) < 5e-4 and
         abs(float(src["因果滚动模型"]) - float(cau.replace(" ", ""))) < 5e-4 and
         abs(float(src["差额（因果−全知）"]) - float(diff.replace(" ", ""))) < 5e-4)
    print(f"  {'OK  ' if m else 'DIFF'} {label}: csv=({src['全知视角（完美预见）']}, {src['因果滚动模型']}, {src['差额（因果−全知）']}, {src['相对差（%）']})")
print("  shares:", 1100899.098134 / 2031467.799512, 930568.701378 / 2031467.799512)

print("== 7.12.4 specified dates ==")
for dt in ("2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"):
    tot = [r for r in spec if r["日期"] == dt and r["指标"].startswith("全天的")][0]["数值"]
    fee = [r for r in spec if r["日期"] == dt and r["指标"].startswith("全天购电费")][0]["数值"]
    e0 = [r for r in daily if r["日期"] == dt][0]["0:00储电量（kWh）"]
    e1 = [r for r in daily if r["日期"] == dt][0]["24:00储电量（kWh）"]
    print(f"  {dt}: real total={float(tot):,.3f} fee={float(fee):,.3f} e0={float(e0):.0f} eN={float(e1):.0f} "
          f"| in doc: {f'{float(tot):,.3f}'.replace(',', ' ') in doc and f'{float(fee):,.3f}'.replace(',', ' ') in doc}")
