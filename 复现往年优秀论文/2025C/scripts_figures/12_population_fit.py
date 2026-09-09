"""补充图；对应PDF p7-8／印刷p11-12；由本次系数计算。
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

NAME = '12_LMM总体效应曲线'
SOURCE = '补充图；对应PDF p7-8／印刷p11-12；由本次系数计算'

def draw(tables, args):
    b=tables('12_固定效应与论文表1对照').set_index('term').estimate
    d=tables('00_男胎清洗数据');t=np.linspace(d.week.min(),d.week.max(),200)
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    for bmi in np.quantile(d.bmi,[.1,.5,.9]):
        mean=b['Intercept']+b['week']*t+b['week_sq']*t*t+b['bmi']*bmi
        ax.plot(t,100*mean,label=f'BMI={bmi:.1f}')
    ax.axhline(4,color='gray',ls='--',lw=1)
    ax.set(xlabel='孕周',ylabel='总体均值Y浓度（%）',title='LMM总体固定效应（身高标准分=0，随机截距=0）');ax.legend()
    return fig

if __name__ == '__main__':
    figure_main(draw, NAME, SOURCE)
