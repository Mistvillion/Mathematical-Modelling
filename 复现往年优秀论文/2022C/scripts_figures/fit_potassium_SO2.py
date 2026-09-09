"""附录B 高钾SO2回归拟合。来源：PDF p35循环；数值对应p14-15表13-16。附录代码会生成但PDF未单列的拟合图；重算。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scripts_tables.plotting import plt, finish, figure_main
from scripts_tables.weathering import fit_trends


# 绘制并保存高钾玻璃SO2的四阶段中心与二次回归曲线
def draw(data, args):
    fits, _ = fit_trends(data)
    r = fits.loc[fits['类型'].eq('高钾') & fits['成分'].eq('SO2')].iloc[0]
    t = np.linspace(1, 4, 120)
    y = np.polyval(r[['a', 'b', 'c']].to_numpy(float), t)
    centers = r[[f'阶段{i}中心' for i in range(1, 5)]].to_numpy(float)
    fig, ax = plt.subplots(figsize=(7.6, 5.2), layout='constrained')
    ax.scatter([1,2,3,4], centers, s=45, color='#a64c38', label='四组中心（非纵向观测）', zorder=3)
    ax.plot(t, y, color='#315b78', label='二次最小二乘拟合')
    r2 = f"{r['R2']:.4f}" if pd.notna(r['R2']) else '未定义（常量）'
    ax.set_title('高钾 · SO2 二次回归（R²=' + r2 + '）')
    ax.set_xticks([1,2,3,4])
    ax.set_xlabel('假设阶段（按表13/14中心顺序；高钾3/4与表11相反）')
    ax.set_ylabel('非零子组成log-ratio中心')
    ax.grid(alpha=.2)
    ax.legend(loc='best', fontsize=9)
    finish(fig, args, 'fit_potassium_SO2')


if __name__ == "__main__":
    figure_main(draw, __doc__)
