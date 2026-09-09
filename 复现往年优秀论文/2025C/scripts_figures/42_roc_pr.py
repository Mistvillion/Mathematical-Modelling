"""补充图；OOF异常概率来自1-P(无异常)。
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

NAME = '42_ROC与PR曲线'
SOURCE = '补充图；OOF异常概率来自1-P(无异常)'

def draw(tables, args):
    from sklearn.metrics import roc_curve,precision_recall_curve,roc_auc_score,average_precision_score
    d=tables('42_女胎逐记录OOF预测');y=d.true_state>0;p=d.p_abnormal
    fpr,tpr,_=roc_curve(y,p);precision,recall,_=precision_recall_curve(y,p)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    axes[0].plot(fpr,tpr,label=f'AUC={roc_auc_score(y,p):.3f}');axes[0].plot([0,1],[0,1],ls='--',color='gray')
    axes[0].set(xlabel='假阳性率',ylabel='真阳性率',title='ROC曲线');axes[0].legend()
    axes[1].plot(recall,precision,label=f'AP={average_precision_score(y,p):.3f}');axes[1].axhline(y.mean(),ls='--',color='gray',label='异常比例基线')
    axes[1].set(xlabel='召回率',ylabel='精确率',title='PR曲线');axes[1].legend()
    return fig

if __name__ == '__main__':
    figure_main(draw, NAME, SOURCE)
