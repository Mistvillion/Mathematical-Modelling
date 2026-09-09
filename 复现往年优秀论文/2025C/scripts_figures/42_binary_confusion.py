"""图21；PDF p33／印刷p38；真正按孕妇分组OOF结果。
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

NAME = '42_异常二分类混淆矩阵'
SOURCE = '图21；PDF p33／印刷p38；真正按孕妇分组OOF结果'

def draw(tables, args):
    d=tables('42_混淆矩阵');d=d[(d.kind=='binary')&(d.variant=='stack')]
    a=d.pivot(index='true',columns='predicted',values='count').to_numpy()
    fig,ax=plt.subplots(figsize=(7,6),layout='constrained');im=ax.imshow(a,cmap='Blues')
    for i in range(2):
        for j in range(2):ax.text(j,i,str(a[i,j]),ha='center',va='center',fontsize=24,color='white' if a[i,j]>a.max()/2 else '#27343D')
    ax.set(xticks=[0,1],xticklabels=['无非整倍体','非整倍体'],yticks=[0,1],yticklabels=['无非整倍体','非整倍体'],xlabel='预测',ylabel='AB真实标签',title='女胎异常判定：按孕妇分组的OOF结果')
    fig.colorbar(im,ax=ax,shrink=.8,label='记录数')
    return fig

if __name__ == '__main__':
    figure_main(draw, NAME, SOURCE)
