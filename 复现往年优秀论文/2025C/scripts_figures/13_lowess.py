"""图11；PDF p10／印刷p14；重绘。
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

NAME = '13_残差LOWESS平滑'
SOURCE = '图11；PDF p10／印刷p14；重绘'

def draw(tables, args):
    d=tables('13_残差')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,col,label in zip(axes,['residual','standardized_residual'],['原始残差','标准化残差']):
        ax.scatter(d.fitted,d[col],s=7,alpha=.25)
        smooth=lowess(d[col],d.fitted,frac=.5)
        ax.plot(smooth[:,0],smooth[:,1],color=COLORS[1],lw=2,label='LOWESS')
        ax.axhline(0,color='gray',ls='--');ax.set(xlabel='拟合值',ylabel=label);ax.legend()
    fig.suptitle('残差与拟合值的平滑关系')
    return fig

if __name__ == '__main__':
    figure_main(draw, NAME, SOURCE)
