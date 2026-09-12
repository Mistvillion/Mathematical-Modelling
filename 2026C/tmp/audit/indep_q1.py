"""Independent re-derivation of Problem 1 (no reuse of the project script).

Reads 附件1 directly, builds the day-ahead storage MILP with a differently
structured formulation, solves it with HiGHS via scipy, and compares against
the numbers written by scripts_tables/1_optimization.py.
"""

from __future__ import annotations

import numpy as np
import openpyxl
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import lil_matrix

DT = 1 / 6
EMIN, EMAX, E0 = 1200.0, 10800.0, 6000.0
PMAX = 5000.0
ETA_C = ETA_D = 0.9
N = 144

wb = openpyxl.load_workbook(
    "CUMCM 2026 C题/附件/附件1.xlsx", read_only=True, data_only=True
)
rows = list(wb["Sheet1"].iter_rows(values_only=True))
hdr, data = rows[0], rows[1:]
print("header:", hdr, "| n rows:", len(data))

price = np.array([float(r[1]) for r in data])
load_p = np.array([float(r[2]) for r in data])
pv_p = np.array([float(r[3]) for r in data])
labels = [str(r[0]) for r in data]
print("time labels first/last:", labels[0], labels[-1], "n =", len(labels))

load_e = load_p * DT
pv_e = pv_p * DT
print(f"daily load  = {load_e.sum():.6f} kWh")
print(f"daily PV    = {pv_e.sum():.6f} kWh")
print(f"load-PV     = {load_e.sum() - pv_e.sum():.6f} kWh")
print(f"price range = {price.min():.6f} .. {price.max():.6f}, mean {price.mean():.6f}")
print(f"PV>0 slots  = {(pv_p > 0).sum()}, max PV power = {pv_p.max():.2f} kW")
print(f"max load    = {load_p.max():.2f} kW, min load = {load_p.min():.2f} kW")

# ---------------- formulation A: user's structure ----------------
def build(eta_c, eta_d, terminal=E0):
    nvar = 6 * N + 1
    o = dict(g=0, c=N, d=2 * N, r=3 * N, u=4 * N, e=5 * N)
    cvec = np.zeros(nvar)
    cvec[o["g"]:o["g"] + N] = price
    lb = np.zeros(nvar)
    ub = np.full(nvar, np.inf)
    ub[o["c"]:o["c"] + N] = PMAX * DT
    ub[o["d"]:o["d"] + N] = PMAX * DT
    ub[o["r"]:o["r"] + N] = pv_e
    ub[o["u"]:o["u"] + N] = 1.0
    lb[o["e"]:] = EMIN
    ub[o["e"]:] = EMAX
    lb[o["e"]] = ub[o["e"]] = E0
    lb[o["e"] + N] = ub[o["e"] + N] = terminal
    integ = np.zeros(nvar, dtype=int)
    integ[o["u"]:o["u"] + N] = 1

    A = lil_matrix((4 * N, nvar))
    lc = np.full(4 * N, -np.inf)
    uc = np.full(4 * N, np.inf)
    for t in range(N):
        A[t, o["g"] + t] = 1
        A[t, o["c"] + t] = -1
        A[t, o["d"] + t] = 1
        A[t, o["r"] + t] = -1
        lc[t] = uc[t] = load_e[t] - pv_e[t]

        row = N + t
        A[row, o["c"] + t] = -eta_c
        A[row, o["d"] + t] = 1 / eta_d
        A[row, o["e"] + t] = -1
        A[row, o["e"] + t + 1] = 1
        lc[row] = uc[row] = 0.0

        A[2 * N + t, o["c"] + t] = 1
        A[2 * N + t, o["u"] + t] = -PMAX * DT
        uc[2 * N + t] = 0.0

        A[3 * N + t, o["d"] + t] = 1
        A[3 * N + t, o["u"] + t] = PMAX * DT
        uc[3 * N + t] = PMAX * DT
    return cvec, lb, ub, integ, A.tocsr(), lc, uc, o


cvec, lb, ub, integ, A, lc, uc, o = build(ETA_C, ETA_D)
res = milp(
    c=cvec, integrality=integ, bounds=Bounds(lb, ub),
    constraints=LinearConstraint(A, lc, uc),
    options={"mip_rel_gap": 1e-9, "disp": False},
)
print("\n=== MILP (user formulation, eta_c=eta_d=0.9) ===")
print(res.message, "| objective =", res.fun)
x = res.x
g, cc, dd, rr, uu, ee = (x[o["g"]:o["g"] + N], x[o["c"]:o["c"] + N],
                          x[o["d"]:o["d"] + N], x[o["r"]:o["r"] + N],
                          x[o["u"]:o["u"] + N], x[o["e"]:])
print("total purchase kWh =", g.sum())
print("sum p*g            =", float(np.dot(price, g)))
print("curtailment kWh    =", rr.sum())
print("max simultaneous   =", float(np.minimum(cc, dd).max()))
print("SOC min/max        =", ee.min(), ee.max(), "| e0,eN =", ee[0], ee[-1])
print("total charge/dis   =", cc.sum(), dd.sum())

# ---------------- formulation B: LP relaxation only (no mode binaries) ------
cvec2, lb2, ub2, integ2, A2, lc2, uc2, o2 = build(ETA_C, ETA_D)
lp = milp(
    c=cvec2, integrality=np.zeros_like(integ2), bounds=Bounds(lb2, ub2),
    constraints=LinearConstraint(A2, lc2, uc2), options={"disp": False},
)
print("\n=== LP relaxation ===")
print(lp.message, "| objective =", lp.fun)
y = lp.x
print("max simultaneous ch/dis in LP =",
      float(np.minimum(y[o2["c"]:o2["c"] + N], y[o2["d"]:o2["d"] + N]).max()))
print("total purchase kWh =", y[o2["g"]:o2["g"] + N].sum(),
      "| curtailment =", y[o2["r"]:o2["r"] + N].sum())

# ---------------- greedy/alternative: long-horizon LP with no-binary,
#                using a "no simultaneous" post-check via price monotonicity --
# Solve an equivalent problem where curtailment is a free sink and verify the
# dual/marginal argument: battery should charge at low price, discharge at high.
order = np.argsort(price)
print("\ncheapest 8 slots (index, label, price):",
      [(int(i), labels[i], round(float(price[i]), 4)) for i in order[:8]])
print("dearest  8 slots (index, label, price):",
      [(int(i), labels[i], round(float(price[i]), 4)) for i in order[-8:]])

# ---------------- sensitivity on efficiency convention ----------
print("\n=== efficiency-convention sensitivity (MILP) ===")
for ec, ed, name in [(0.9, 0.9, "charge 0.9 / discharge 0.9 (user)"),
                     (0.9, 1.0, "charge 0.9 / discharge 1.0"),
                     (1.0, 0.9, "charge 1.0 / discharge 0.9"),
                     (0.9486832980505138, 0.9486832980505138, "round-trip 0.9")]:
    cv, l, u, ig, AA, ll, uu2, oo = build(ec, ed)
    r2 = milp(c=cv, integrality=ig, bounds=Bounds(l, u),
              constraints=LinearConstraint(AA, ll, uu2),
              options={"mip_rel_gap": 1e-9, "disp": False})
    print(f"  {name:34s} cost = {r2.fun:12.6f}  "
          f"purchase = {r2.x[oo['g']:oo['g']+N].sum():12.6f}")

# ---------------- start-label vs end-label alignment sensitivity ----------
print("\n=== time-alignment sensitivity (shift price/load/PV by one slot) ===")
# convention (B): sample labelled X covers [X, X+10) -> the first slot of the
# calendar day has NO data, the last data row spills past 24:00.
price_s = np.roll(price, -1)   # slot t uses the sample labelled t*10+10 ... 
load_s = np.roll(load_e, -1)
pv_s = np.roll(pv_e, -1)
cvec3 = np.zeros(6 * N + 1)
cvec3[o["g"]:o["g"] + N] = price_s
u3 = np.full(6 * N + 1, np.inf)
u3[o["r"]:o["r"] + N] = pv_s
u3[o["c"]:o["c"] + N] = PMAX * DT
u3[o["d"]:o["d"] + N] = PMAX * DT
u3[o["u"]:o["u"] + N] = 1.0
l3 = np.zeros(6 * N + 1)
l3[o["e"]:] = EMIN
u3[o["e"]:] = EMAX
l3[o["e"]] = u3[o["e"]] = E0
l3[o["e"] + N] = u3[o["e"] + N] = E0
A3 = lil_matrix((4 * N, 6 * N + 1))
lc3 = np.full(4 * N, -np.inf)
uc3 = np.full(4 * N, np.inf)
for t in range(N):
    A3[t, o["g"] + t] = 1
    A3[t, o["c"] + t] = -1
    A3[t, o["d"] + t] = 1
    A3[t, o["r"] + t] = -1
    lc3[t] = uc3[t] = load_s[t] - pv_s[t]
    r_ = N + t
    A3[r_, o["c"] + t] = -ETA_C
    A3[r_, o["d"] + t] = 1 / ETA_D
    A3[r_, o["e"] + t] = -1
    A3[r_, o["e"] + t + 1] = 1
    lc3[r_] = uc3[r_] = 0.0
    A3[2 * N + t, o["c"] + t] = 1
    A3[2 * N + t, o["u"] + t] = -PMAX * DT
    uc3[2 * N + t] = 0.0
    A3[3 * N + t, o["d"] + t] = 1
    A3[3 * N + t, o["u"] + t] = PMAX * DT
    uc3[3 * N + t] = PMAX * DT
r3 = milp(c=cvec3, integrality=integ, bounds=Bounds(l3, u3),
          constraints=LinearConstraint(A3.tocsr(), lc3, uc3),
          options={"mip_rel_gap": 1e-9, "disp": False})
print("  shifted-by-one-slot optimum cost =", r3.fun)
