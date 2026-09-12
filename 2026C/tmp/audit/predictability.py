import numpy as np, openpyxl
from datetime import date
DT=1/6
wb=openpyxl.load_workbook("CUMCM 2026 C题/附件/附件2.xlsx",read_only=True,data_only=True)
L=np.array([[float(v) for v in r[1:]] for r in list(wb['小区负载'].iter_rows(values_only=True))[1:]])
P=np.array([[float(v) for v in r[1:]] for r in list(wb['光伏发电实际功率'].iter_rows(values_only=True))[1:]])
wb.close()
P[P<1.0]=0.0
d=[date.fromordinal(date(2025,1,1).toordinal()+i) for i in range(365)]
wd=np.array([x.weekday() for x in d])
names=['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
print("daily mean load by weekday (kW):")
for k in range(7):
    m=L[wd==k].mean(axis=1)
    print(f"  {names[k]}: daily mean {m.mean():8.1f}  |  daily total {L[wd==k].sum(axis=1).mean()/6:10.1f} kWh")
print("\n0.8-quantile / median / 0.2-quantile of daily total load by weekday (kWh):")
for k in range(7):
    tot=L[wd==k].sum(axis=1)/6
    print(f"  {names[k]}: q20={np.quantile(tot,.2):9.0f} q50={np.quantile(tot,.5):9.0f} q80={np.quantile(tot,.8):9.0f}")
print("\ndaily PV total by weekday, monthly mean (kWh):")
for k in range(7):
    tot=P[wd==k].sum(axis=1)/6
    print(f"  {names[k]}: mean {tot.mean():9.0f}")
print("\nmonthly means:")
import collections
mo=np.array([x.month for x in d])
for m in range(1,13):
    s=mo==m
    print(f"  {m:2d}: load/day {L[s].sum(axis=1).mean()/6:9.0f} kWh | pv/day {P[s].sum(axis=1).mean()/6:9.0f} kWh | "
          f"load mean {L[s].mean():7.1f} kW | pv mean {P[s].mean():7.1f} kW")
print("\n--- intrinsic spread of per-slot load within a weekday class ---")
for k in range(7):
    x=L[wd==k]            # (n_days, 144)
    m=np.median(x,axis=0)
    print(f"  {names[k]}: n={x.shape[0]} median abs dev from class-median curve = {np.abs(x-m).mean():7.2f} kW "
          f"({100*np.abs(x-m).mean()/m.mean():.1f}% of mean)")
print("\n--- PV: within-month spread ---")
for m in (1,4,7,10):
    s=mo==m
    x=P[s]; med=np.median(x,axis=0)
    print(f"  month {m:2d}: n={x.shape[0]} MAD from month-median curve = {np.abs(x-med).mean():7.2f} kW ({100*np.abs(x-med).mean()/med.mean():.1f}%)")
