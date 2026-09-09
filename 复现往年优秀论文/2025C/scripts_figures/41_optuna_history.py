"""图20；PDF p65／印刷p36；本次搜索记录。
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

NAME = '41_专家Optuna搜索过程'
SOURCE = '图20；PDF p65／印刷p36；本次搜索记录'

def draw(tables, args):
    d=tables('41_Optuna搜索记录')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,name in zip(axes,['t13','t21']):
        for fold,g in d[d.model==name].groupby('outer_fold'):
            ax.plot(g.trial,g.log_loss.cummin(),marker='.',label=f'外层折{fold}')
        ax.set(xlabel='Optuna试验编号',ylabel='历史最佳内层LogLoss',title=f'{name.upper()} 专家');ax.legend()
    return fig

if __name__ == '__main__':
    figure_main(draw, NAME, SOURCE)
