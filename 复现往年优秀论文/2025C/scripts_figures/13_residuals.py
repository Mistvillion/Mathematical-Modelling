"""图10；PDF p10／印刷p14；标准化残差按残差标准差计算。
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

NAME = '13_残差四联图'
SOURCE = '图10；PDF p10／印刷p14；标准化残差按残差标准差计算'

def draw(tables, args):
    d=tables('13_残差')
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for ax,col,absolute,label in [(axes[0,0],'residual',False,'原始残差'),(axes[0,1],'standardized_residual',False,'标准化残差'),(axes[1,0],'residual',True,'残差绝对值'),(axes[1,1],'standardized_residual',True,'标准化残差绝对值')]:
        y=d[col].abs() if absolute else d[col]
        ax.scatter(d.fitted,y,s=7,alpha=.4);ax.axhline(0,color=COLORS[1],ls='--',lw=1)
        ax.set(xlabel='包含随机截距的拟合值',ylabel=label)
    fig.suptitle('LMM残差诊断')
    return fig

if __name__ == '__main__':
    figure_main(draw, NAME, SOURCE)
