"""图8；PDF p64／印刷p10；重算。
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

NAME = '11_Y浓度相关强度'
SOURCE = '图8；PDF p64／印刷p10；重算'

def draw(tables, args):
    d=tables('11_Pearson相关').sort_values('r')
    fig,ax=plt.subplots(figsize=(9,5.6),layout='constrained')
    ax.barh([FEATURE_NAMES.get(c,c) for c in d.feature],d.r,color=[COLORS[1] if r<0 else COLORS[0] for r in d.r])
    ax.axvline(0,color='#56616B',lw=.8);ax.set(xlabel='Pearson r',title='各指标与Y染色体浓度的记录层相关性')
    return fig

if __name__ == '__main__':
    figure_main(draw, NAME, SOURCE)
