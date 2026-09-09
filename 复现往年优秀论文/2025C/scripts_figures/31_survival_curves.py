"""补充图；两种模型测试集预测。
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

NAME = '31_测试孕妇达标概率曲线'
SOURCE = '补充图；两种模型测试集预测'

def draw(tables, args):
    d=tables('31_个体达标概率');d=d[d.split=='test']
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,(mode,g) in zip(axes,d.groupby('mode',sort=False)):
        for pid in g.person_id.unique()[:12]:
            p=g[g.person_id==pid];ax.plot(p.week,p.cdf,alpha=.35,lw=1)
        mean=g.groupby('week').cdf.mean();ax.plot(mean.index,mean,color='black',lw=2,label='54人平均')
        ax.set(xlabel='孕周',ylabel='模型累计达标概率',ylim=(0,1.02),title='完整轨迹回顾重构' if mode=='paper_history' else '静态区间删失对照');ax.legend()
    return fig

if __name__ == '__main__':
    figure_main(draw, NAME, SOURCE)
