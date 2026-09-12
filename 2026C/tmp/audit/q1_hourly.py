import csv, numpy as np
rows=list(csv.DictReader(open('outputs/tables/1_完整计划购电策略.csv',encoding='utf-8-sig')))
def c(n): return np.array([float(r[n]) for r in rows])
p=c('电价（元/kWh）'); g=c('计划购电量（kWh）'); ch=c('充电量（kWh）'); di=c('放电量（kWh）')
ee=c('时段初储电量（kWh）'); eN=float(rows[-1]['时段末储电量（kWh）']); v=c('可用光伏电量（kWh）'); l=c('负载电量（kWh）')
soc=np.r_[ee,eN]
print(f"{'hour':>5s} {'price':>6s} {'load':>8s} {'pv':>8s} {'buy':>8s} {'chg':>8s} {'dis':>8s} {'SOCend':>9s}")
for h in range(24):
    s=slice(h*6,h*6+6)
    print(f"{h:5d} {p[s].mean():6.3f} {l[s].sum():8.0f} {v[s].sum():8.0f} {g[s].sum():8.0f} {ch[s].sum():8.0f} {di[s].sum():8.0f} {soc[min(h*6+6,144)]:9.0f}")
print(f"\ntotals: buy {g.sum():.2f} kWh, cost {(p*g).sum():.2f} 元, charge {ch.sum():.2f}, discharge {di.sum():.2f}, losses {0.1*ch.sum()+(1/0.9-1)*di.sum():.2f}")
print(f"hours with buy==0: {sum(1 for h in range(24) if g[h*6:(h+1)*6].sum()<1e-6)}")
print(f"hours with curtailment>0: {int((c('弃光量（kWh）')>1e-9).sum())}")
print(f"max hourly buy {max(g[h*6:(h+1)*6].sum() for h in range(24)):.1f}")
