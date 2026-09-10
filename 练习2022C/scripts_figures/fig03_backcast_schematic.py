"""图3 预测示意图。来源：PDF p16。概念示意重绘；示例曲线不代表附件拟合结果。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scripts_tables.plotting import plt, finish, figure_main


# 绘制并保存通过曲线平移反推风化前成分的示意图
def draw(data, args):
    t = np.linspace(.6, 4.2, 240)
    f = lambda x: .24*x*x - .18*x + 1.6
    t_obs, y_obs = 3.4, 2.6
    d = f(t_obs) - y_obs
    fig, ax = plt.subplots(figsize=(8, 5.4), layout='constrained')
    ax.plot(t, f(t), color='#315b78', label='示例拟合曲线 f(t)')
    ax.plot(t, f(t)-d, color='#cf9050', label='平移曲线 f(t)-d')
    ax.scatter([1, t_obs], [f(1)-d, y_obs], color='#a33d50', zorder=3)
    ax.axvline(1, color='#a33d50', linestyle='--', alpha=.6)
    ax.annotate('', (t_obs, f(t_obs)), (t_obs, y_obs), arrowprops={'arrowstyle': '<->'})
    ax.annotate('d = f(t*) - y*', (t_obs+.08, (f(t_obs)+y_obs)/2))
    ax.annotate('待预测点 (t*, y*)', (t_obs, y_obs), xytext=(-110,-30), textcoords='offset points')
    ax.annotate('反推值 f(1)-d', (1, f(1)-d), xytext=(10,-25), textcoords='offset points')
    ax.set(xlabel='假设阶段 t（不是实测时间）', ylabel='某成分的log-ratio坐标',
           title='曲线平移的逆向预测思路（概念示意）')
    ax.legend(loc='upper left')
    ax.grid(alpha=.2)
    finish(fig, args, 'fig03_backcast_schematic')


if __name__ == "__main__":
    figure_main(draw, __doc__)
