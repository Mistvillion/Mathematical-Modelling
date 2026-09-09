"""图5；PDF p62／印刷p8；按题目GC阈值重算。
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

NAME = '00_检测质量分布'
SOURCE = '图5；PDF p62／印刷p8；按题目GC阈值重算'

def draw(tables, args):
    d=pd.concat([tables('00_男胎清洗数据'),tables('00_女胎清洗数据')])
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    axes[0].hist(d.gc*100,bins=35,color=COLORS[0]);axes[0].axvline(40,color=COLORS[1],ls='--')
    axes[0].set(xlabel='GC含量（%）',ylabel='记录数',title='GC含量分布')
    counts=[d.gc.between(.4,.6).sum(),(~d.gc.between(.4,.6)).sum()]
    axes[1].bar(['40%-60%内','范围外'],counts,color=COLORS[:2])
    axes[1].set(ylabel='记录数',title='题目阈值质控标记')
    for i,n in enumerate(counts):axes[1].text(i,n,f'{n/len(d):.1%}',ha='center',va='bottom')
    axes[2].bar(['T13','T18','T21'],[d.t13.sum(),d.t18.sum(),d.t21.sum()],color=COLORS[:3])
    axes[2].set(ylabel='记录数（可重叠）',title='AB列非整倍体分布')
    return fig

if __name__ == '__main__':
    figure_main(draw, NAME, SOURCE)
