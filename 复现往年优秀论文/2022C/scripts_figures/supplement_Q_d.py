"""补充 铅钡风化Q型亚类树。来源：对应PDF p22-23。新增：图7成员划分对应的Ward树。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scripts_tables.plotting import plt, finish, figure_main
from scipy.cluster.hierarchy import linkage, dendrogram
from scripts_tables.subclasses import subclass_result


# 绘制并保存铅钡风化样本基于论文代表成分的Q型亚类聚类树
def draw(data, args):
    frame, features, x, labels = subclass_result(data, '铅钡', '风化')
    fig, ax = plt.subplots(figsize=(8.2, max(4.6, len(frame)*.22)), layout='constrained')
    dendrogram(linkage(x, method='ward'), labels=frame['文物采样点'].tolist(), orientation='right',
               leaf_font_size=8, color_threshold=0, above_threshold_color='#315b78', ax=ax)
    ax.set_title('铅钡风化：Q型亚类聚类\n原文表25特征：' + ', '.join(features))
    ax.set_xlabel('Ward 合并距离（原始百分比，未标准化）')
    finish(fig, args, 'supplement_Q_d')


if __name__ == "__main__":
    figure_main(draw, __doc__)
