"""补充图；对应PDF p12-14／印刷p16-18；实际拟合结果。
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

NAME = '21_个体GPR曲线示例'
SOURCE = '补充图；对应PDF p12-14／印刷p16-18；实际拟合结果'

def draw(tables, args):
    curves=tables('21_GPR预测曲线');obs=tables('00_男胎清洗数据')
    pids=curves.person_id.unique();chosen=pids[np.linspace(0,len(pids)-1,4).astype(int)]
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for ax,pid in zip(axes.flat,chosen):
        d=curves[curves.person_id==pid];g=obs[obs.person_id==pid]
        ax.fill_between(d.week,100*(d['mean']-1.96*d.sd),100*(d['mean']+1.96*d.sd),alpha=.15)
        ax.plot(d.week,100*d['mean'],label='GPR均值');ax.plot(d.week,100*d.lower95,ls='--',label='95%预测下界')
        ax.scatter(g.week,g.y*100,color=COLORS[1],s=22,zorder=4,label='实测')
        ax.axhline(4,color='gray',ls=':');ax.set(title=f'孕妇{pid}',xlabel='孕周',ylabel='Y浓度（%）',xlim=(10,40));ax.legend(fontsize=8)
    fig.suptitle('个体GPR与预测区间（含外推部分）')
    return fig

if __name__ == '__main__':
    figure_main(draw, NAME, SOURCE)
