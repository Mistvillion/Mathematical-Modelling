import numpy as np, openpyxl, csv
DT=1/6
def actual(sheet):
    wb=openpyxl.load_workbook("CUMCM 2026 C题/附件/附件2.xlsx",read_only=True,data_only=True)
    rows=list(wb[sheet].iter_rows(values_only=True)); wb.close()
    return [r[0].date().isoformat() for r in rows[1:]], np.array([[float(v) for v in r[1:]] for r in rows[1:]])*DT
d,load=actual('小区负载'); _,pv=actual('光伏发电实际功率'); pv[pv<DT]=0
price=np.array([float(r[1]) for r in list(openpyxl.load_workbook("CUMCM 2026 C题/附件/附件1.xlsx",read_only=True,data_only=True)['Sheet1'].iter_rows(values_only=True))[1:]])
p=np.tile(price,334); sel=[d.index(x) for x in sorted(set(d))[31:]]
lo=np.concatenate([load[i] for i in sel]); vo=np.concatenate([pv[i] for i in sel])
net=np.maximum(lo-vo,0)
print("official period: load %.3f kWh, pv %.3f kWh, net %.3f kWh"%(lo.sum(),vo.sum(),(lo-vo).sum()))
print("no-storage + perfect net-load forecast  : %.6f 元"%(p@net))
print("no-storage + 0.8-quantile forecast plan : 18291991.421036 元 (computed earlier)")
print("storage    + perfect information (dyn)  : 12244796.854916 元")
print("storage    + perfect foresight LP bound : 12229817.388753 元")
print("storage    + causal forecast (delivered): 14272696.316983 元")
print()
print("Q1 (附件1 day): load %.3f kWh, pv %.3f kWh, net %.3f kWh"%(
    (np.array([float(r[2]) for r in list(openpyxl.load_workbook("CUMCM 2026 C题/附件/附件1.xlsx",read_only=True,data_only=True)['Sheet1'].iter_rows(values_only=True))[1:]])*DT).sum(),
    (np.array([float(r[3]) for r in list(openpyxl.load_workbook("CUMCM 2026 C题/附件/附件1.xlsx",read_only=True,data_only=True)['Sheet1'].iter_rows(values_only=True))[1:]])*DT).sum(),
    (np.array([float(r[2])-float(r[3]) for r in list(openpyxl.load_workbook("CUMCM 2026 C题/附件/附件1.xlsx",read_only=True,data_only=True)['Sheet1'].iter_rows(values_only=True))[1:]])*DT).sum()))
