"""图1；PDF p4／印刷p4；按本复现实现重绘。
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

NAME = '00_总体思路框架'
SOURCE = '图1；PDF p4／印刷p4；按本复现实现重绘'

def draw(tables, args):
    fig, ax = plt.subplots(figsize=(12, 7), layout='constrained')
    nodes = [(.5,.9,.65,.08,'原始附件：男胎1082条／267人，女胎605条／147人'),
             (.5,.73,.6,.08,'数据校验、孕周转换、BMI填补、保留重复检测与删失'),
             (.15,.51,.25,.16,'问题一\nPearson + 随机截距LMM\nREML估计／ML比较'),
             (.5,.51,.27,.16,'问题二、三\nGPR／MLP+LSTM+DeepHit\nBMI分组与时点优化'),
             (.85,.51,.25,.16,'问题四\n4个LightGBM专家\nOOF堆叠 + Z规则'),
             (.5,.22,.7,.12,'验证：按孕妇划分、GA与DP对照、读数误差模拟\n输出：CSV、Excel、独立图形、论文差异说明')]
    flow(ax,nodes,[(0,1),(1,2),(1,3),(1,4),(2,5),(3,5),(4,5)])
    ax.set_title('C132 模型复现与验证流程',pad=18)
    return fig

if __name__ == '__main__':
    figure_main(draw, NAME, SOURCE)
