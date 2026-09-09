"""补充图；对应原文表10；按孕妇分组OOF。
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

NAME = '42_具体异常分类混淆矩阵'
SOURCE = '补充图；对应原文表10；按孕妇分组OOF'

def draw(tables, args):
    d=tables('42_混淆矩阵');d=d[(d.kind=='multiclass')&(d.variant=='stack')]
    a=d.pivot(index='true',columns='predicted',values='count').to_numpy()[:7,:7]
    fig,ax=plt.subplots(figsize=(10,8),layout='constrained');im=ax.imshow(np.log1p(a),cmap='Blues')
    for i in range(7):
        for j in range(7):ax.text(j,i,str(a[i,j]),ha='center',va='center',color='white' if np.log1p(a[i,j])>np.log1p(a.max())*.6 else '#27343D')
    labels=[LABELS[i] for i in range(7)]
    ax.set(xticks=range(7),xticklabels=labels,yticks=range(7),yticklabels=labels,xlabel='预测',ylabel='AB真实标签',title='具体异常分类（颜色为log(1+记录数)，文字为原始计数）')
    plt.setp(ax.get_xticklabels(),rotation=35,ha='right')
    return fig

if __name__ == '__main__':
    figure_main(draw, NAME, SOURCE)
