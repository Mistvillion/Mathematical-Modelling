"""图15；PDF p23／印刷p27；结构示意。
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

NAME = '31_MLP静态编码器'
SOURCE = '图15；PDF p23／印刷p27；结构示意'

def draw(tables, args):
    fig,ax=plt.subplots(figsize=(9,6),layout='constrained')
    nodes=[(.5,.85,.7,.12,'4维输入：年龄、身高、首次体重、首次BMI\n标准化参数只在训练孕妇上估计'),(.5,.57,.65,.12,'全连接4→32\nReLU + Dropout(0.4)'),(.5,.29,.65,.12,'全连接32→16\nReLU + Dropout(0.4)'),(.5,.07,.55,.07,'16维静态特征表示')]
    flow(ax,nodes,[(0,1),(1,2),(2,3)]);ax.set_title('MLP静态特征编码器')
    return fig

if __name__ == '__main__':
    figure_main(draw, NAME, SOURCE)
