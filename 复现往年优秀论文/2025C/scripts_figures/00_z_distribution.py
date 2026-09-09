"""图3；PDF p61／印刷p7；按AB异常与无异常分组重绘。
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

NAME = '00_染色体Z值分布'
SOURCE = '图3；PDF p61／印刷p7；按AB异常与无异常分组重绘'

def draw(tables, args):
    d=pd.concat([tables('00_男胎清洗数据'),tables('00_女胎清洗数据')])
    fig,axes=plt.subplots(1,4,figsize=(12,4.5),layout='constrained')
    for ax,col in zip(axes,['z13','z18','z21','zx']):
        ax.boxplot([d.loc[d.abnormal==v,col].dropna() for v in [0,1]],tick_labels=['AB无异常','AB异常'],showfliers=False)
        ax.axhline(3,color=COLORS[1],ls='--',lw=1)
        ax.axhline(-3,color=COLORS[1],ls='--',lw=1)
        ax.set(title=FEATURE_NAMES[col],ylabel='Z值')
    fig.suptitle('染色体Z值分布（隐藏离群点仅为显示，计算保留全部记录）')
    return fig

if __name__ == '__main__':
    figure_main(draw, NAME, SOURCE)
