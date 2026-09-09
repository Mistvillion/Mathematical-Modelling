"""图14；PDF p20／印刷p24；本次GA三组结果。
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

NAME = '22_GPR达标比例响应'
SOURCE = '图14；PDF p20／印刷p24；本次GA三组结果'

def draw(tables, args):
    d=tables('22_BMI分组策略');d=d[(d.solver=='GA')&(d.k==3)]
    fig,ax=plt.subplots(figsize=(10,5.5),layout='constrained')
    ax.axhspan(10,12,color=COLORS[2],alpha=.08);ax.axhspan(12,27,color=COLORS[1],alpha=.05)
    for group,g in d.groupby('group'):
        ax.plot(g.target,g.week,marker='o',label=['低BMI组','中BMI组','高BMI组'][group-1])
    ax.axhline(12,color='gray',ls='--',lw=1)
    ax.set(xlabel='要求的组内达标比例（%）',ylabel='推荐孕周',title='GPR + GA：每个目标比例重新优化BMI边界');ax.legend()
    return fig

if __name__ == '__main__':
    figure_main(draw, NAME, SOURCE)
