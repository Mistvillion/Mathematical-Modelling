"""补充图；同时报告准确率与少数类检出能力。
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

NAME = '42_模型与基线比较'
SOURCE = '补充图；同时报告准确率与少数类检出能力'

def draw(tables, args):
    d=tables('42_总体与分折指标');d=d[(d.scope=='OOF_all')&(d.variant.isin(['stack','all_normal_baseline','z3_baseline']))]
    fig,ax=plt.subplots(figsize=(11,5),layout='constrained');x=np.arange(3)
    for j,col in enumerate(['accuracy','balanced_accuracy','recall']):
        ax.bar(x+(j-1)*.25,d[col],.25,label={'accuracy':'准确率','balanced_accuracy':'平衡准确率','recall':'异常召回率'}[col])
    ax.set(xticks=x,xticklabels=['堆叠模型','全部预测无异常','Z≥3规则'],ylim=(0,1),ylabel='指标值',title='类别不平衡下的OOF模型表现');ax.legend()
    return fig

if __name__ == '__main__':
    figure_main(draw, NAME, SOURCE)
