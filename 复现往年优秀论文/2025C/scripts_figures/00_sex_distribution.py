"""图4；PDF p62／印刷p8；重绘。
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

NAME = '00_男女胎生理指标分布'
SOURCE = '图4；PDF p62／印刷p8；重绘'

def draw(tables, args):
    m,f=tables('00_男胎清洗数据'),tables('00_女胎清洗数据')
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for ax,col in zip(axes.flat,['age','bmi','height','weight']):
        ax.hist(m[col],bins=20,density=True,alpha=.55,label='男胎')
        ax.hist(f[col],bins=20,density=True,alpha=.55,label='女胎')
        ax.set(xlabel=FEATURE_NAMES[col],ylabel='密度')
        ax.legend()
    fig.suptitle('男女胎孕妇特征比较（检测记录层）')
    return fig

if __name__ == '__main__':
    figure_main(draw, NAME, SOURCE)
