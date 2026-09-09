"""图16；PDF p25／印刷p29；显示本次mask及尾部概率修正。
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

NAME = '31_混合生存网络结构'
SOURCE = '图16；PDF p25／印刷p29；显示本次mask及尾部概率修正'

def draw(tables, args):
    fig,ax=plt.subplots(figsize=(11,8),layout='constrained')
    nodes=[(.23,.87,.36,.1,'4维基线生理特征'),(.77,.87,.38,.1,'变长(孕周，Y浓度)序列'),(.23,.66,.36,.12,'MLP：4→32→16\nReLU + Dropout'),(.77,.66,.38,.12,'LSTM：2→16\n按有效长度打包，忽略填充'),(.5,.43,.7,.1,'拼接32维 → 全连接32维 → 输出32个logits'),(.5,.2,.75,.14,'Softmax：10-40周的PMF + 40周以后尾部\n累计求和得到达标CDF\n似然包含事件和删失；排序项用于GPR伪标签模式')]
    flow(ax,nodes,[(0,2),(1,3),(2,4),(3,4),(4,5)]);ax.set_title('MLP + LSTM + DeepHit（修复后实现）')
    return fig

if __name__ == '__main__':
    figure_main(draw, NAME, SOURCE)
