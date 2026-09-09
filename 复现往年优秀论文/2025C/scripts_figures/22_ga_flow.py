"""图13；PDF p18／印刷p22；GA和补充DP审计流程。
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

NAME = '22_遗传算法流程'
SOURCE = '图13；PDF p18／印刷p22；GA和补充DP审计流程'

def draw(tables, args):
    fig,ax=plt.subplots(figsize=(9,7),layout='constrained')
    nodes=[(.5,.91,.6,.07,'选择达标比例P和分组数k；孕妇按BMI排序'),(.5,.73,.65,.08,'初始化BMI分割点种群，禁止拆分相同BMI'),(.5,.54,.68,.1,'区间达标时间 → 分段风险 × 组内人数\n违反最小组规模：惩罚'),(.5,.33,.65,.1,'锦标赛选择 → 均匀交叉 → BMI点变异\n记录历史最佳个体'),(.5,.12,.7,.1,'输出GA方案；另用动态规划求同一目标的全局最优\n报告风险差，区分相同风险下的多种分法')]
    flow(ax,nodes,[(0,1),(1,2),(2,3),(3,4)]);ax.set_title('BMI分组的遗传搜索与最优性审计')
    return fig

if __name__ == '__main__':
    figure_main(draw, NAME, SOURCE)
