"""图2；PDF p61／印刷p7；重算并区分AB与出生结局。
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

NAME = '00_年龄BMI与异常率'
SOURCE = '图2；PDF p61／印刷p7；重算并区分AB与出生结局'

def draw(tables, args):
    d = pd.concat([tables('00_男胎清洗数据'),tables('00_女胎清洗数据')])
    fig, axes = plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    for ax,col,bins,labels in [(axes[0],'age',[0,25,30,35,100],['≤25','26-30','31-35','≥36']),
                               (axes[1],'bmi',[0,25,30,35,40,np.inf],['<25','25-30','30-35','35-40','≥40'])]:
        groups = pd.cut(d[col],bins=bins,labels=labels,right=col=='age')
        summary=d.groupby(groups,observed=False)[['abnormal','birth_unhealthy']].mean()*100
        x=np.arange(len(summary))
        ax.bar(x-.19,summary.abnormal,.38,label='AB列非整倍体')
        ax.bar(x+.19,summary.birth_unhealthy,.38,label='出生后不健康')
        ax.set(xticks=x,xticklabels=labels,ylabel='检测记录比例（%）',xlabel=FEATURE_NAMES[col])
        ax.legend(fontsize=8)
    fig.suptitle('年龄、BMI与两类结局的描述关系（记录层，非因果效应）')
    return fig

if __name__ == '__main__':
    figure_main(draw, NAME, SOURCE)
