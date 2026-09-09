"""图7b；PDF p64／印刷p10；精简到四个核心变量重绘。
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

NAME = '11_核心变量散点矩阵'
SOURCE = '图7b；PDF p64／印刷p10；精简到四个核心变量重绘'

def draw(tables, args):
    d=tables('00_男胎清洗数据')[['y','week','bmi','height']].rename(columns=FEATURE_NAMES)
    axes=pd.plotting.scatter_matrix(d,figsize=(10,10),diagonal='hist',alpha=.25,s=6,color=COLORS[0],hist_kwds={'bins':20})
    fig=axes[0,0].figure
    fig.suptitle('核心变量关系矩阵',y=1.01)
    for ax in axes.flat: ax.tick_params(labelsize=7)
    fig.subplots_adjust(wspace=.08,hspace=.08)
    return fig

if __name__ == '__main__':
    figure_main(draw, NAME, SOURCE)
