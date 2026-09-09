"""补充图；固定网络及分组，只扰动有效Y输入。
单张图入口；修改本文件的draw即可调整布局。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess
from scripts_tables.plotting import plt, COLORS, flow, figure_main
from scripts_tables.constants import FEATURE_NAMES, LABELS

NAME = '33_深度网络读数误差敏感性'
SOURCE = '补充图；固定网络及分组，只扰动有效Y输入'

def draw(tables, args):
    d=tables('33_读数误差汇总');target=90 if 90 in d.target.values else int(d.target.max());d=d[d.target==target]
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    for group,g in d.groupby('group'):
        ax.errorbar(g.sigma_y*100,g.delta_mean*7,yerr=g.week_sd.fillna(0)*7,marker='o',capsize=3,label=f'第{group}组')
    ax.axhline(0,color='gray',ls='--');ax.set(xlabel='Y读数噪声标准差（百分点）',ylabel='推荐时间变化（天）',title=f'完整轨迹模型：{target}%目标下的条件误差传播');ax.legend()
    return fig

if __name__ == '__main__':
    figure_main(draw, NAME, SOURCE)
