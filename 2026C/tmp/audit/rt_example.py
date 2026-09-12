"""Print a slot-by-slot comparison of frozen-storage vs real-time-storage on one day."""
import csv, numpy as np, openpyxl
from scipy.optimize import linprog
from scipy.sparse import lil_matrix
DT=1/6; EMIN,EMAX,EREF=1200.,10800.,6000.; ETC=ETD=0.9; LIM=5000/6; N=144; PEN=5.0; LAM=1.3952
TARGET = "2025-09-23"
def actual(sheet):
    wb=openpyxl.load_workbook("CUMCM 2026 C题/附件/附件2.xlsx",read_only=True,data_only=True)
    rows=list(wb[sheet].iter_rows(values_only=True)); wb.close()
    return [r[0].date().isoformat() for r in rows[1:]], np.array([[float(v) for v in r[1:]] for r in rows[1:]])*DT
d,load=actual('小区负载'); _,pv=actual('光伏发电实际功率'); pv[pv<DT]=0
price=np.array([float(r[1]) for r in list(openpyxl.load_workbook("CUMCM 2026 C题/附件/附件1.xlsx",read_only=True,data_only=True)['Sheet1'].iter_rows(values_only=True))[1:]])
det=list(csv.DictReader(open('outputs/tables/2_滚动预测与调度明细.csv',encoding='utf-8-sig')))
k=[det[i*N]['日期'] for i in range(334)].index(TARGET)
blk=det[k*N:(k+1)*N]
f=lambda n: np.array([float(r[n]) for r in blk])
g=f('计划购电量（kWh）'); dp=f('计划放电量（kWh）'); cp=f('计划充电量（kWh）')
da=f('实际放电量（kWh）'); ca=f('实际充电量（kWh）'); h=f('紧急购电量（kWh）'); e0=f('时段初储电量（kWh）')
i=d.index(TARGET); la,va=load[i],pv[i]

# --- real-time battery dispatch with frozen purchase (same LP as bench_rt_battery) ---
nv=6*N+3; o=dict(h=0,c=N,d=2*N,r=3*N,w=4*N,e=5*N); zp,zm=6*N+1,6*N+2
cvec=np.zeros(nv); cvec[o['h']:o['h']+N]=PEN*price; cvec[zp]=cvec[zm]=LAM
lb=np.zeros(nv); ub=np.full(nv,np.inf)
ub[o['c']:o['c']+N]=LIM; ub[o['d']:o['d']+N]=LIM; ub[o['r']:o['r']+N]=va; ub[o['w']:o['w']+N]=g
lb[o['e']:o['e']+N+1]=EMIN; ub[o['e']:o['e']+N+1]=EMAX; lb[o['e']]=ub[o['e']]=e0[0]
A=lil_matrix((2*N+1,nv)); lo=np.full(2*N+1,-np.inf); hi=np.full(2*N+1,np.inf)
for t in range(N):
    A[t,o['h']+t]=1; A[t,o['c']+t]=-1; A[t,o['d']+t]=1; A[t,o['r']+t]=-1; A[t,o['w']+t]=-1
    lo[t]=hi[t]=la[t]-va[t]-g[t]
    r=N+t; A[r,o['c']+t]=-ETC; A[r,o['d']+t]=1/ETD; A[r,o['e']+t]=-1; A[r,o['e']+t+1]=1; lo[r]=hi[r]=0.
A[2*N,o['e']+N]=1; A[2*N,zp]=-1; A[2*N,zm]=1; lo[2*N]=hi[2*N]=EREF
res=linprog(cvec,A_eq=A.tocsr(),b_eq=lo,bounds=np.column_stack((lb,ub)),method='highs')
assert res.success,res.message
x=res.x; d_rt=x[o['d']:o['d']+N]; h_rt=x[o['h']:o['h']+N]; c_rt=x[o['c']:o['c']+N]
print(f"day = {TARGET}")
print(f"frozen  : emergency {h.sum():8.3f} kWh, cost {PEN*float(price@h):9.2f} 元, discharge {da.sum():9.3f}, charge {ca.sum():9.3f}")
print(f"real-time: emergency {h_rt.sum():8.3f} kWh, cost {PEN*float(price@h_rt):9.2f} 元, discharge {d_rt.sum():9.3f}, charge {c_rt.sum():9.3f}")
print(f"plan    : discharge {dp.sum():9.3f}, charge {cp.sum():9.3f}")
print(f"\nslots where the two executions differ (t, interval, price, plan_d, frozen_d, rt_d, frozen_h, rt_h):")
print(f"{'#':>4s} {'interval':13s} {'price':>6s} {'plan_d':>9s} {'froz_d':>9s} {'rt_d':>9s} {'froz_h':>9s} {'rt_h':>9s}")
for t in range(N):
    if abs(da[t]-d_rt[t])>1e-6 or abs(h[t]-h_rt[t])>1e-6 or abs(ca[t]-c_rt[t])>1e-6:
        print(f"{t:4d} {t*10//60:02d}:{t*10%60:02d}-{(t+1)*10//60:02d}:{(t+1)*10%60:02d} {price[t]:6.3f} {dp[t]:9.3f} {da[t]:9.3f} {d_rt[t]:9.3f} {h[t]:9.3f} {h_rt[t]:9.3f}")
print(f"\nSOC at 24:00: frozen {e0[0]+ETC*ca.sum()-da.sum()/ETD:.3f} | real-time {float(x[o['e']+N]):.3f} | plan target {EREF:.3f}")
