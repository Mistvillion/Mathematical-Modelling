"""Cross-check the four specified-date summaries against result1/result2 workbooks."""
import csv, numpy as np, openpyxl
# ---- Q1 ----
wb=openpyxl.load_workbook("outputs/results/result1.xlsx",read_only=True)
rows=list(wb["计划购电量"].iter_rows(values_only=True))[1:]
lab=[r[0] for r in rows]; val=np.array([float(r[1]) for r in rows])
want={10:0.0,12:480.412450,14:0.0,16:445.431650,18:531.894050,20:0.0}
print("Q1 result1.xlsx: value at slot index h*6 (=interval [h:00,h:10)) vs summary csv")
for h,v in want.items():
    print(f"  {h:02d}:00-{h:02d}:10  xlsx row label '{lab[h*6]:12s}' value {val[h*6]:12.6f}  csv {v:12.6f}  match={abs(val[h*6]-v)<5e-7}")
print(f"  daily total from xlsx = {val.sum():.6f} (csv/report 59482.698998)")

# ---- Q2 ----
wb=openpyxl.load_workbook("outputs/results/result2.xlsx",read_only=True)
ws=wb["计划购电量"]
rows=list(ws.iter_rows(values_only=True))
hdr=rows[0]; body=rows[1:]
dates=[r[0].date().isoformat() for r in body]
plans=np.array([[float(v) for v in r[1:145]] for r in body])
tot=np.array([float(r[145]) for r in body]); fee=np.array([float(r[146]) for r in body])
summ={}
for r in csv.DictReader(open("outputs/tables/2_指定日期购电量及全天结果.csv",encoding="utf-8-sig")):
    summ.setdefault(r["日期"],[]).append(r)
print("\nQ2 result2.xlsx vs 2_指定日期购电量及全天结果.csv")
for d,items in summ.items():
    i=dates.index(d)
    for r in items:
        if r["指标"]=="指定时段计划购电量":
            h=int(r["时间段"][:2]); v=float(r["数值"])
            slot=h*6
            ok=abs(plans[i,slot]-v)<5e-7
            print(f"  {d} {r['时间段']} xlsx col '{hdr[1+slot]:12s}' = {plans[i,slot]:12.6f} | summary {v:12.6f} | match={ok}")
        elif r["指标"]=="全天的计划购电量":
            print(f"  {d} 全天的计划购电量 xlsx total={tot[i]:.6f} | summary {float(r['数值']):.6f} | row-sum diff={abs(tot[i]-plans[i].sum()):.1e}")
        else:
            print(f"  {d} 全天购电费 xlsx={fee[i]:.6f} | summary {float(r['数值']):.6f} (含紧急购电)")
# check the xlsx fee == plan cost + emergency cost
det=list(csv.DictReader(open("outputs/tables/2_滚动预测与调度明细.csv",encoding="utf-8-sig")))
pc=np.array([float(r["时段计划购电费（元）"]) for r in det]).reshape(334,144).sum(axis=1)
ec=np.array([float(r["时段紧急购电费（元）"]) for r in det]).reshape(334,144).sum(axis=1)
print("  xlsx fee   == plan cost + emergency cost ?", np.abs(fee-(pc+ec)).max())
print("  xlsx total == sum of the 144 plan cells ?", np.abs(tot-plans.sum(axis=1)).max())
