"""Check the Q2 deliverable workbook against the template and the CSVs."""

from __future__ import annotations

import numpy as np
import openpyxl

TEMPLATE = "CUMCM 2026 C题/附件/附件5/result2.xlsx"
OUT = "outputs/results/result2.xlsx"

tpl = openpyxl.load_workbook(TEMPLATE, read_only=True, data_only=True)
out = openpyxl.load_workbook(OUT, read_only=True, data_only=True)
print("sheets template:", tpl.sheetnames)
print("sheets output  :", out.sheetnames)

# headers identical?
tpl_hdr = [c for c in next(tpl["计划购电量"].iter_rows(values_only=True))]
out_hdr = [c for c in next(out["计划购电量"].iter_rows(values_only=True))]
print("\nheader length:", len(tpl_hdr), len(out_hdr))
print("headers identical:", tpl_hdr == out_hdr)
print("first 3:", out_hdr[:3])
print("last 4 :", out_hdr[-4:])
diff = [(i, a, b) for i, (a, b) in enumerate(zip(tpl_hdr, out_hdr)) if a != b]
print("header diffs:", diff)

ps = out["计划购电量"]
rows = list(ps.iter_rows(values_only=True))
print("\n计划购电量 rows:", len(rows), "(1 + 334 expected)")
body = rows[1:]
print("first date:", body[0][0], "last date:", body[-1][0])
vals = np.array([[float(v) for v in r[1:145]] for r in body])
print("plan block shape:", vals.shape)
print("finite:", np.isfinite(vals).all(), "| negative values:", int((vals < 0).sum()))
tot = np.array([float(r[145]) for r in body])
fee = np.array([float(r[146]) for r in body])
print("全天购电量 sum:", tot.sum(), "| column sum check:", np.abs(tot - vals.sum(axis=1)).max())
print("全天购电费 sum:", fee.sum())
print("全天购电量 vs 购电费 all positive:", (tot >= 0).all(), (fee >= 0).all())
print("days with 全天购电量 != row sum (tol 1e-6):",
      int((np.abs(tot - vals.sum(axis=1)) > 1e-6).sum()))

# storage sheet
ss = out["充放电量"]
srows = list(ss.iter_rows(values_only=True))
print("\n充放电量 rows:", len(srows), "(1 + 334*6 = 2005 expected)")
print("header:", srows[0])
print("row1:", srows[1])
print("row6:", srows[6])
print("row7:", srows[7])
print("last:", srows[-1])
blocks = np.array([[float(r[2]) if r[2] is not None else 0.0,
                    float(r[3]) if r[3] is not None else 0.0] for r in srows[1:]])
print("charge sum:", blocks[:, 0].sum(), "| discharge sum:", blocks[:, 1].sum())
print("any negative:", (blocks < 0).any())
soc = [r[5] for r in srows[1:] if r[5] is not None]
print("SOC entries:", len(soc), "min", min(soc), "max", max(soc))
labels = set(r[1] for r in srows[1:])
print("block labels:", labels)

# emergency sheet
es = out["紧急购电量"]
erows = list(es.iter_rows(values_only=True))
print("\n紧急购电量 rows:", len(erows))
print("header:", erows[0])
for r in erows[1:6]:
    print("  ", r)
print("  ...")
for r in erows[-3:]:
    print("  ", r)
energies = np.array([float(r[2]) for r in erows[1:] if r[2] is not None])
print("n intervals:", len(energies), "| sum:", energies.sum(), "| min:", energies.min())
dates_e = [r[0] for r in erows[1:] if r[0] is not None]
print("n distinct dates:", len(set(dates_e)))
