"""补充图；固定原GA分组，扰动实测Y，未重新拟合GPR。
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

NAME = '23_读数误差与实际达标覆盖率'
SOURCE = '补充图；固定原GA分组，扰动实测Y，未重新拟合GPR'

def draw(tables, args):
    d=tables('23_读数误差汇总');target=90 if 90 in d.target.values else int(d.target.max());d=d[d.target==target]
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    for group,g in d.groupby('group'):
        ax.errorbar(g.sigma_y*100,g.coverage_mean*100,yerr=g.coverage_sd.fillna(0)*100,marker='o',capsize=4,label=f'第{group}组')
    ax.axhline(target,color='gray',ls='--');ax.set(xlabel='Y读数噪声标准差（百分点）',ylabel='在原推荐时点的实测达标比例（%）',title=f'{target}%目标下的实测误差敏感性（包含全部267人）');ax.legend()
    return fig

if __name__ == '__main__':
    figure_main(draw, NAME, SOURCE)
