"""Q1: sensitivity to the still-ambiguous convention choices."""
import numpy as np, openpyxl
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import lil_matrix
DT=1/6; EMIN,EMAX,E0=1200.,10800.,6000.; P=5000.; EC=ED=0.9; N=144
rows=list(openpyxl.load_workbook("CUMCM 2026 C题/附件/附件1.xlsx",read_only=True,data_only=True)["Sheet1"].iter_rows(values_only=True))[1:]
price=np.array([float(r[1]) for r in rows]); L=np.array([float(r[2]) for r in rows])*DT; V=np.array([float(r[3]) for r in rows])*DT

def solve(lim_c, lim_d, eta_c=EC, eta_d=ED, terminal=None):
    n=6*N+1; o=dict(g=0,c=N,d=2*N,r=3*N,u=4*N,e=5*N)
    c=np.zeros(n); c[o['g']:o['g']+N]=price
    lb=np.zeros(n); ub=np.full(n,np.inf)
    ub[o['c']:o['c']+N]=lim_c; ub[o['d']:o['d']+N]=lim_d; ub[o['r']:o['r']+N]=V
    ub[o['u']:o['u']+N]=1.0; lb[o['e']:]=EMIN; ub[o['e']:]=EMAX
    lb[o['e']]=ub[o['e']]=E0
    lb[o['e']+N]=ub[o['e']+N]=E0 if terminal is None else terminal
    A=lil_matrix((4*N,n)); lo=np.full(4*N,-np.inf); hi=np.full(4*N,np.inf)
    for t in range(N):
        A[t,o['g']+t]=1; A[t,o['c']+t]=-1; A[t,o['d']+t]=1; A[t,o['r']+t]=-1
        lo[t]=hi[t]=L[t]-V[t]
        r=N+t; A[r,o['c']+t]=-eta_c; A[r,o['d']+t]=1/eta_d; A[r,o['e']+t]=-1; A[r,o['e']+t+1]=1
        lo[r]=hi[r]=0.
        A[2*N+t,o['c']+t]=1; A[2*N+t,o['u']+t]=-lim_c; hi[2*N+t]=0.
        A[3*N+t,o['d']+t]=1; A[3*N+t,o['u']+t]=lim_d; hi[3*N+t]=lim_d
    integ=np.zeros(n,dtype=int); integ[o['u']:o['u']+N]=1
    r_=milp(c=c,integrality=integ,bounds=Bounds(lb,ub),constraints=LinearConstraint(A.tocsr(),lo,hi),
            options={"mip_rel_gap":1e-9,"disp":False})
    return r_.fun, r_.x[o['g']:o['g']+N].sum()

base=solve(P*DT,P*DT)
print(f"{'convention':52s} {'cost 元':>14s} {'purchase kWh':>14s}")
print(f"{'A bus-side limit |ch|,|dis| <= 833.33 (paper)':52s} {base[0]:14.6f} {base[1]:14.6f}")
v=solve(P*DT/EC,P*DT/ED); print(f"{'B battery-side limit (eta*c <= 833.33, d/eta <= 833.33)':52s} {v[0]:14.6f} {v[1]:14.6f}")
v=solve(P*DT/EC,P*DT);     print(f"{'C charge limited on battery side only':52s} {v[0]:14.6f} {v[1]:14.6f}")
v=solve(P*DT,P*DT/ED);     print(f"{'D discharge limited on battery side only':52s} {v[0]:14.6f} {v[1]:14.6f}")
v=solve(P*DT,P*DT,terminal=EMAX); print(f"{'E terminal SOC free at 10800 instead of 6000':52s} {v[0]:14.6f} {v[1]:14.6f}")
v=solve(P*DT,P*DT,terminal=EMIN); print(f"{'F terminal SOC = 1200':52s} {v[0]:14.6f} {v[1]:14.6f}")
# efficiency conventions
for name,(ec,ed) in {"G charge 0.9 / discharge 1.0":(0.9,1.0),"H charge 1.0 / discharge 0.9":(1.0,0.9),
                      "I round-trip 0.9 (0.9487 each)":(np.sqrt(0.9),np.sqrt(0.9)),
                      "J round-trip 0.81 (0.9 each)":(0.9,0.9)}.items():
    v=solve(P*DT,P*DT,eta_c=ec,eta_d=ed); print(f"{name:52s} {v[0]:14.6f} {v[1]:14.6f}")
# curtailment fully disabled (PV surplus must be stored or wasted) == same
# no-storage baseline
ns=np.maximum(L-V,0.); print(f"\n{'no-storage baseline (buy net load per slot)':52s} {float(price@ns):14.6f} {ns.sum():14.6f}")
