"""图18；PDF p28／印刷p32；本次完整轨迹模型GA结果。
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

NAME = '32_深度生存达标比例响应'
SOURCE = '图18；PDF p28／印刷p32；本次完整轨迹模型GA结果'

def draw(tables, args):
    d=tables('32_BMI分组策略');d=d[(d.solver=='GA')&(d['mode']=='paper_history')]
    fig,ax=plt.subplots(figsize=(10,5.5),layout='constrained')
    for group,g in d.groupby('group'):
        ax.plot(g.target,g.week.replace(np.inf,np.nan),marker='o',label=f'第{group}组')
    ax.axhline(12,color='gray',ls='--');ax.set(xlabel='模型平均达标概率要求（%）',ylabel='推荐孕周（向上取整到天）',title='完整轨迹生存模型 + GA（回顾性示范）');ax.legend()
    return fig

if __name__ == '__main__':
    figure_main(draw, NAME, SOURCE)
