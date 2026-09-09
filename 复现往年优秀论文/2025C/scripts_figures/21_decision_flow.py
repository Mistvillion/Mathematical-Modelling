"""图12；PDF p13／印刷p17；按附录B.2重绘。
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

NAME = '21_GPR三阶段决策流程'
SOURCE = '图12；PDF p13／印刷p17；按附录B.2重绘'

def draw(tables, args):
    fig,ax=plt.subplots(figsize=(10,8),layout='constrained')
    nodes=[(.5,.92,.55,.07,'按孕妇聚合：(孕周，Y浓度)序列'),(.24,.72,.34,.1,'单点观测\n≥4%取实测孕周；否则∞'),(.76,.72,.34,.1,'多点观测\n检查时间与浓度相关系数'),(.76,.49,.4,.12,'非负趋势：拟合GPR\n在10-40周查找均值−1.96SD≥4%\n负趋势或未找到：暂记∞'),(.5,.25,.68,.12,'实测修正：取模型时间与最早实测达标时间的较小值\n有限值最低10周；仍未达标者保留∞'),(.5,.06,.65,.06,'输出267人；第二问仅优化有限值，第三问保留删失')]
    flow(ax,nodes,[(0,1),(0,2),(2,3),(1,4),(3,4),(4,5)])
    ax.set_title('GPR三阶段达标时间构造规则')
    return fig

if __name__ == '__main__':
    figure_main(draw, NAME, SOURCE)
