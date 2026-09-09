"""图7a；PDF p64／印刷p10；重绘。
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

NAME = '11_Pearson相关热图'
SOURCE = '图7a；PDF p64／印刷p10；重绘'

def draw(tables, args):
    d=tables('11_相关矩阵').set_index('feature')
    fig,ax=plt.subplots(figsize=(10,8),layout='constrained')
    image=ax.imshow(d,vmin=-1,vmax=1,cmap='RdBu_r')
    labels=[FEATURE_NAMES.get(x,x) for x in d.index]
    ax.set(xticks=range(len(d)),xticklabels=labels,yticks=range(len(d)),yticklabels=labels,title='Pearson相关矩阵（记录层描述）')
    plt.setp(ax.get_xticklabels(),rotation=45,ha='right')
    for i in range(len(d)):
        for j in range(len(d)):
            value=d.iloc[i,j];ax.text(j,i,f'{value:.2f}',ha='center',va='center',fontsize=8,color='white' if abs(value)>.6 else '#27343D')
    fig.colorbar(image,ax=ax,shrink=.8,label='r')
    return fig

if __name__ == '__main__':
    figure_main(draw, NAME, SOURCE)
