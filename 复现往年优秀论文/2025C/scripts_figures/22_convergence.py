"""补充图；真实GA运行轨迹。
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

NAME = '22_GA风险收敛'
SOURCE = '补充图；真实GA运行轨迹'

def draw(tables, args):
    d=tables('22_GA收敛');d=d[d.k==3]
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    for target,g in d.groupby('target'):
        ax.plot(g.generation,g.best_risk,label=f'{target}%')
    ax.set(xlabel='迭代代数',ylabel='历史最小风险',title='三组GA的风险收敛轨迹');ax.legend()
    return fig

if __name__ == '__main__':
    figure_main(draw, NAME, SOURCE)
