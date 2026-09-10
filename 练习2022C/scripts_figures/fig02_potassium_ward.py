"""图2 高钾Q型聚类。来源：PDF p12。根据正文与附录B补写，原始百分比Ward。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scripts_tables.plotting import plt, finish, figure_main
from scipy.cluster.hierarchy import linkage, dendrogram


# 绘制并保存高钾采样点的Ward聚类树
def draw(data, args):
    frame = data.samples.loc[data.samples['类型'].eq('高钾')]
    z = linkage(data.matrix(frame), method='ward')
    fig, ax = plt.subplots(figsize=(9.0, max(5, len(frame)*.21)), layout='constrained')
    dendrogram(z, labels=frame['文物采样点'].tolist(), orientation='right',
               leaf_font_size=8, color_threshold=0, above_threshold_color='#315b78', ax=ax)
    ax.set_xlabel('Ward 合并距离（原始百分比，未标准化）')
    ax.set_title('高钾玻璃：采样点聚类')
    ax.grid(axis='x', alpha=.2)
    finish(fig, args, 'fig02_potassium_ward')


if __name__ == "__main__":
    figure_main(draw, __doc__)
