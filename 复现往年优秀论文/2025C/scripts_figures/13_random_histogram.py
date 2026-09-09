"""图9b；PDF p9／印刷p13；重绘。
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

NAME = '13_随机截距分布'
SOURCE = '图9b；PDF p9／印刷p13；重绘'

def draw(tables, args):
    u=tables('13_随机截距').random_intercept
    fig,ax=plt.subplots(figsize=(7,5),layout='constrained')
    ax.hist(u,bins=24,density=True,alpha=.65)
    x=np.linspace(u.min(),u.max(),200);ax.plot(x,stats.norm.pdf(x,u.mean(),u.std()),color=COLORS[1],label='同均值方差正态密度')
    ax.set(xlabel='估计随机截距',ylabel='密度',title='随机截距分布');ax.legend()
    return fig

if __name__ == '__main__':
    figure_main(draw, NAME, SOURCE)
