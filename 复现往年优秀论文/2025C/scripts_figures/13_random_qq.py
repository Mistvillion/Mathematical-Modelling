"""图9a；PDF p9／印刷p13；每位孕妇仅取一个BLUP。
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

NAME = '13_随机截距QQ图'
SOURCE = '图9a；PDF p9／印刷p13；每位孕妇仅取一个BLUP'

def draw(tables, args):
    u=tables('13_随机截距').random_intercept
    fig,ax=plt.subplots(figsize=(6,5),layout='constrained')
    stats.probplot(u,dist='norm',plot=ax)
    ax.set(title='随机截距QQ图（267位孕妇）',xlabel='标准正态理论分位数',ylabel='估计随机截距')
    return fig

if __name__ == '__main__':
    figure_main(draw, NAME, SOURCE)
