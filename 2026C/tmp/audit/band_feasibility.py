"""How often is the hard daily end-SOC band [5400,6600] reachable at all?

For every official day we take the *published* frozen purchase plan (audited
causal run) and the *actual* load/PV, then compute the highest end-of-day SOC the
battery could possibly reach (best case: all chargeable surplus is stored, no
discharge needed at all). If that upper bound is below 5400 the day is
structurally infeasible under a hard daily band.
"""
import csv, numpy as np, openpyxl
DT=1/6; ETC=0.9; ETD=0.9; LIM=5000/6; N=144; EMIN,EMAX=1200.,10800.
def actual(sheet):
    wb=openpyxl.load_workbook("CUMCM 2026 C题/附件/附件2.xlsx",read_only=True,data_only=True)
    rows=list(wb[sheet].iter_rows(values_only=True)); wb.close()
    return [r[0].date().isoformat() for r in rows[1:]], np.array([[float(v) for v in r[1:]] for r in rows[1:]])*DT
d,load=actual('小区负载'); _,pv=actual('光伏发电实际功率'); pv[pv<DT]=0
det=list(csv.DictReader(open('outputs/tables/2_滚动预测与调度明细.csv',encoding='utf-8-sig')))
assert len(det)==334*N
days=[det[k*N]['日期'] for k in range(334)]
g=np.array([float(r['计划购电量（kWh）']) for r in det]).reshape(334,N)
e0=np.array([float(r['时段初储电量（kWh）']) for r in det]).reshape(334,N)[:,0]
# also the *forecast-based* reachability (what the controller could have known)
lp=np.array([float(r['预测负载电量（kWh）']) for r in det]).reshape(334,N)
vp=np.array([float(r['预测光伏电量（kWh）']) for r in det]).reshape(334,N)

rows=[]
for k,dd in enumerate(days):
    i=d.index(dd); la,va=load[i],pv[i]
    cap_act=np.minimum(LIM,np.maximum(g[k]+va-la,0.0))
    reach_act=min(EMAX,e0[k]+ETC*cap_act.sum())
    cap_fc=np.minimum(LIM,np.maximum(g[k]+vp[k]-lp[k],0.0))
    reach_fc=min(EMAX,e0[k]+ETC*cap_fc.sum())
    rows.append((dd,e0[k],reach_act,reach_fc,cap_act.sum(),cap_fc.sum()))
A=np.array([[r[2],r[3]] for r in rows])
bad_a=int((A[:,0]<5400-1e-6).sum()); bad_f=int((A[:,1]<5400-1e-6).sum())
print(f"days whose reachable end SOC < 5400  (actual data): {bad_a}/334")
print(f"days whose reachable end SOC < 5400  (0:00 forecast): {bad_f}/334")
print(f"min reachable end SOC: actual {A[:,0].min():.1f} ({rows[int(np.argmin(A[:,0]))][0]}), forecast {A[:,1].min():.1f}")
print(f"slack (reachable - 5400): actual p1={np.percentile(A[:,0]-5400,1):.1f}, p5={np.percentile(A[:,0]-5400,5):.1f}, median={np.median(A[:,0]-5400):.1f}")
print("\nworst 12 days under a hard daily band (actual):")
for dd,e,ra,rf,ca,cf in sorted(rows,key=lambda r:r[2])[:12]:
    print(f"  {dd}  e0={e:8.1f}  chargeable(actual)={ca:8.1f} -> max end SOC={ra:8.1f}  "
          f"(forecast-side {rf:8.1f})  {'INFEASIBLE' if ra<5400 else ''}")
