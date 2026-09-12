import csv, numpy as np
from collections import defaultdict
rows=list(csv.DictReader(open('outputs/tables/2_每日运行审计.csv',encoding='utf-8-sig')))
mon=defaultdict(list)
for r in rows:
    mon[r['日期'][:7]].append((float(r['日初储电量']),float(r['实际日末储电量']),float(r['紧急购电量']),float(r['实际总购电费']),float(r['未利用计划购电量']),float(r['实际弃光量'])))
print(f"{'month':8s} {'SOCbegin min..max':>24s} {'SOCend mean':>11s} {'emerg kWh':>10s} {'cost 元':>13s} {'unused kWh':>10s} {'curt kWh':>11s}")
for m in sorted(mon):
    v=mon[m]; e0=[x[0] for x in v]; e1=[x[1] for x in v]
    print(f"{m:8s} {min(e0):10.1f}..{max(e0):9.1f} {np.mean(e1):11.1f} {sum(x[2] for x in v):10.1f} {sum(x[3] for x in v):13.1f} {sum(x[4] for x in v):10.1f} {sum(x[5] for x in v):11.1f}")
