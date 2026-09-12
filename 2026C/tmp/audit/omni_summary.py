import csv, numpy as np
rows=list(csv.DictReader(open("outputs/tables/2_全知视角逐日费用.csv",encoding="utf-8-sig")))
print("daily rows",len(rows))
k=list(rows[0].keys()); print("cols:",k)
g=np.array([float(r['计划购电量（kWh）']) for r in rows]); c=np.array([float(r['购电费（元）']) for r in rows])
e0=np.array([float(r['0:00储电量（kWh）']) for r in rows]); eN=np.array([float(r['24:00储电量（kWh）']) for r in rows])
cur=np.array([float(r['弃光量（kWh）']) for r in rows]); ch=np.array([float(r['充电量（kWh）']) for r in rows]); di=np.array([float(r['放电量（kWh）']) for r in rows])
print(f"purchase {g.sum():.3f} kWh | cost {c.sum():.6f} 元 | curtail {cur.sum():.3f} | chg {ch.sum():.1f} | dis {di.sum():.1f}")
print(f"e0 range {e0.min():.1f}..{e0.max():.1f} | eN range {eN.min():.1f}..{eN.max():.1f}")
for v in (5400.0,6600.0):
    print(f"  days with eN == {v}: {int((np.abs(eN-v)<1e-6).sum())}")
print(f"  eN strictly inside: {int(((eN>5400+1e-6)&(eN<6600-1e-6)).sum())}")
print(f"  days with 弃光>0: {int((cur>1e-6).sum())}")
print(f"  daily cost: min {c.min():.1f} ({rows[int(c.argmin())]['日期']}) max {c.max():.1f} ({rows[int(c.argmax())]['日期']}) mean {c.mean():.1f}")
print(f"  daily purchase: min {g.min():.1f} max {g.max():.1f}")
mm={}
for r in rows: mm.setdefault(r['日期'][:7],[0.0,0.0,0.0,0.0])
for r in rows:
    m=r['日期'][:7]; v=mm[m]
    v[0]+=float(r['计划购电量（kWh）']); v[1]+=float(r['购电费（元）']); v[2]+=float(r['弃光量（kWh）']); v[3]+=float(r['充电量（kWh）'])
print("\nmonth  purchase kWh      cost 元       curt kWh      chg kWh")
for m in sorted(mm): print(f"{m} {mm[m][0]:15.1f} {mm[m][1]:13.1f} {mm[m][2]:13.1f} {mm[m][3]:13.1f}")
