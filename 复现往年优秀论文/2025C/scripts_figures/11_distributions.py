"""图6；PDF p63／印刷p9；重绘。
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

NAME = '11_核心变量分布'
SOURCE = '图6；PDF p63／印刷p9；重绘'

def draw(tables, args):
    d=tables('00_男胎清洗数据')
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for ax,col in zip(axes.flat,['y','week','bmi','height']):
        values=d[col]*100 if col=='y' else d[col]
        ax.hist(values,bins=26,color=COLORS[0],edgecolor='white',linewidth=.5)
        ax.set(xlabel='Y浓度（%）' if col=='y' else FEATURE_NAMES[col],ylabel='记录数')
    fig.suptitle('男胎1082条检测记录的核心变量分布')
    return fig

if __name__ == '__main__':
    figure_main(draw, NAME, SOURCE)
