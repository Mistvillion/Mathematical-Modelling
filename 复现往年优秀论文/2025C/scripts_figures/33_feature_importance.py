"""附录B.3补充图；PDF p46／印刷p51；测试集10次排列。
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

NAME = '33_全部特征排列重要性'
SOURCE = '附录B.3补充图；PDF p46／印刷p51；测试集10次排列'

def draw(tables, args):
    d=tables('33_排列重要性');d=d[d['mode']=='paper_history'].sort_values('loss_increase_mean')
    labels=[FEATURE_NAMES.get(x,'完整时序信息') for x in d.feature]
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    ax.barh(labels,d.loss_increase_mean,xerr=d.loss_increase_sd,capsize=3)
    ax.axvline(0,color='gray',lw=1);ax.set(xlabel='测试损失增加量（均值±标准差）',title='完整轨迹重构模型：排列重要性')
    return fig

if __name__ == '__main__':
    figure_main(draw, NAME, SOURCE)
