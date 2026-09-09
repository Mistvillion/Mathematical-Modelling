"""图17；PDF p27／印刷p31；本次训练记录。
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

NAME = '31_生存网络训练损失'
SOURCE = '图17；PDF p27／印刷p31；本次训练记录'

def draw(tables, args):
    d=tables('31_训练损失')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,(mode,g) in zip(axes,d.groupby('mode',sort=False)):
        ax.plot(g.epoch,g.train_loss,label='训练');ax.plot(g.epoch,g.validation_loss,label='验证')
        ax.axvline(g.loc[g.validation_loss.idxmin(),'epoch'],color='gray',ls=':',label='最佳检查点')
        ax.set(xlabel='训练轮数',ylabel='损失',title='完整轨迹回顾重构' if mode=='paper_history' else '静态区间删失对照');ax.legend()
    fig.suptitle('训练集与验证集按孕妇独立划分；两种损失数值不可横向比较')
    return fig

if __name__ == '__main__':
    figure_main(draw, NAME, SOURCE)
