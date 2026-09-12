"""Independently verify the new 全知视角 (perfect-foresight) model."""
import numpy as np, openpyxl
from scipy.optimize import linprog
from scipy.sparse import lil_matrix
DT=1/6; EMIN,EMAX=1200.,10800.; ETC=ETD=0.9; LIM=5000/6; N=144
def actual(sheet):
    wb=openpyxl.load_workbook("CUMCM 2026 C题/附件/附件2.xlsx",read_only=True,data_only=True)
    rows=list(wb[sheet].iter_rows(values_only=True)); wb.close()
    return [r[0].date().isoformat() for r in rows[1:]], np.array([[float(v) for v in r[1:]] for r in rows[1:]])*DT
d,load=actual('小区负载'); _,pv=actual('光伏发电实际功率'); pv[pv<DT]=0
price=np.array([float(r[1]) for r in list(openpyxl.load_workbook("CUMCM 2026 C题/附件/附件1.xlsx",read_only=True,data_only=True)['Sheet1'].iter_rows(values_only=True))[1:]])
days=sorted(set(d))[31:]                      # 2025-02-01 .. 12-31
sel=np.array([d.index(x) for x in days]); nD=len(sel); T=nD*N
lo_=np.concatenate([load[i] for i in sel]); vo_=np.concatenate([pv[i] for i in sel]); p_=np.tile(price,nD)
print("days",nD,"T",T,"load",lo_.sum(),"pv",vo_.sum())

def solve(e0, daily_band, label):
    o=dict(g=0,c=T,d=2*T,r=3*T,e=4*T); nv=5*T+1
    cv=np.zeros(nv); cv[o['g']:o['g']+T]=p_
    lb=np.zeros(nv); ub=np.full(nv,np.inf)
    ub[o['c']:o['c']+T]=LIM; ub[o['d']:o['d']+T]=LIM; ub[o['r']:o['r']+T]=vo_
    lb[o['e']:o['e']+T+1]=EMIN; ub[o['e']:o['e']+T+1]=EMAX
    lb[o['e']]=ub[o['e']]=e0
    A=lil_matrix((2*T,nv)); lo=np.full(2*T,-np.inf); hi=np.full(2*T,np.inf)
    for t in range(T):
        A[t,o['g']+t]=1; A[t,o['c']+t]=-1; A[t,o['d']+t]=1; A[t,o['r']+t]=-1
        lo[t]=hi[t]=lo_[t]-vo_[t]
        r=T+t; A[r,o['c']+t]=-ETC; A[r,o['d']+t]=1/ETD; A[r,o['e']+t]=-1; A[r,o['e']+t+1]=1
        lo[r]=hi[r]=0.
    res=linprog(cv,A_eq=A.tocsr(),b_eq=lo,bounds=np.column_stack((lb,ub)),method='highs')
    x=res.x; g=x[o['g']:o['g']+T]; c=x[o['c']:o['c']+T]; dd=x[o['d']:o['d']+T]; r_=x[o['r']:o['r']+T]; e=x[o['e']:o['e']+T+1]
    eN=e[N::N][:nD]
    print(f"{label:34s} obj={res.fun:14.6f} purchase={g.sum():14.3f} curt={r_.sum():12.3f} "
          f"chg={c.sum():12.1f} dis={dd.sum():12.1f} sim={float(np.minimum(c,dd).max()):.3f}")
    print(f"{'':34s} daily end SOC: min={eN.min():9.2f} max={eN.max():9.2f} | 6000-day count={int((np.abs(eN-6000)<1e-6).sum())}/{nD}")
    return res.fun

a=solve(6000.0, False, "band only on Dec-31 (my LP)")
# daily band variant: implement via bounds on each day's last boundary
def solve_daily(e0):
    o=dict(g=0,c=T,d=2*T,r=3*T,e=4*T); nv=5*T+1
    cv=np.zeros(nv); cv[o['g']:o['g']+T]=p_
    lb=np.zeros(nv); ub=np.full(nv,np.inf)
    ub[o['c']:o['c']+T]=LIM; ub[o['d']:o['d']+T]=LIM; ub[o['r']:o['r']+T]=vo_
    lb[o['e']:o['e']+T+1]=EMIN; ub[o['e']:o['e']+T+1]=EMAX
    for k in range(nD):
        lb[o['e']+(k+1)*N]=5400.0; ub[o['e']+(k+1)*N]=6600.0
    lb[o['e']]=ub[o['e']]=e0
    A=lil_matrix((2*T,nv)); lo=np.full(2*T,-np.inf); hi=np.full(2*T,np.inf)
    for t in range(T):
        A[t,o['g']+t]=1; A[t,o['c']+t]=-1; A[t,o['d']+t]=1; A[t,o['r']+t]=-1
        lo[t]=hi[t]=lo_[t]-vo_[t]
        r=T+t; A[r,o['c']+t]=-ETC; A[r,o['d']+t]=1/ETD; A[r,o['e']+t]=-1; A[r,o['e']+t+1]=1
        lo[r]=hi[r]=0.
    res=linprog(cv,A_eq=A.tocsr(),b_eq=lo,bounds=np.column_stack((lb,ub)),method='highs')
    x=res.x; g=x[o['g']:o['g']+T]; c=x[o['c']:o['c']+T]; dd=x[o['d']:o['d']+T]; r_=x[o['r']:o['r']+T]; e=x[o['e']:o['e']+T+1]
    eN=e[N::N][:nD]
    print(f"{'daily band [5400,6600] every day':34s} obj={res.fun:14.6f} purchase={g.sum():14.3f} curt={r_.sum():12.3f} "
          f"chg={c.sum():12.1f} dis={dd.sum():12.1f} sim={float(np.minimum(c,dd).max()):.3f}")
    print(f"{'':34s} daily end SOC: min={eN.min():9.2f} max={eN.max():9.2f} | 6000-day count={int((np.abs(eN-6000)<1e-6).sum())}/{nD}")
    return res.fun
b=solve_daily(6000.0)
print(f"\nuser's 全知视角 total = 12241228.517471  | my daily-band LP = {b:.6f} | diff = {b-12241228.517471:.6f}")
