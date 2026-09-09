"""补充图；核验正文k=3最优的陈述。
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

NAME = '22_分组数量风险对照'
SOURCE = '补充图；核验正文k=3最优的陈述'

def draw(tables, args):
    d=tables('22_GA与动态规划对照')
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    for target,g in d.groupby('target'):
        ax.plot(g.k,g.dp_risk,marker='o',label=f'{target}%（DP）')
    ax.set(xticks=[3,4,5,6],xlabel='分组数k',ylabel='加权风险总和',title='不同分组数的全局最优风险（重合表示并列）');ax.legend()
    return fig

if __name__ == '__main__':
    figure_main(draw, NAME, SOURCE)
