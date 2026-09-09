"""图19；PDF p30／印刷p34；本次计算，允许负重要性。
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

NAME = '33_静态指标排列重要性'
SOURCE = '图19；PDF p30／印刷p34；本次计算，允许负重要性'

def draw(tables, args):
    d=tables('33_排列重要性');d=d[(d['mode']=='paper_history')&(d.feature!='longitudinal')].sort_values('loss_increase_mean')
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    ax.barh([FEATURE_NAMES[x] for x in d.feature],d.loss_increase_mean,xerr=d.loss_increase_sd,capsize=3,color=COLORS[2])
    ax.axvline(0,color='gray',lw=1);ax.set(xlabel='测试损失增加量（均值±标准差）',title='静态生理指标重要性（完整轨迹模型）')
    return fig

if __name__ == '__main__':
    figure_main(draw, NAME, SOURCE)
