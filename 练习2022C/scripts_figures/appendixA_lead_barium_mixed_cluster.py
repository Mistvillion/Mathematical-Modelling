"""附录A 铅钡原始与预测混合聚类。来源：PDF p30-31（高钾图10）。平均联接重算；星号为本包预测；仅相似性对照。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scripts_tables.plotting import plt, finish, figure_main
from scipy.cluster.hierarchy import dendrogram
from scripts_tables.weathering import mixed_cluster_check


# 绘制并保存铅钡原始样本与预测样本的混合聚类树
def draw(data, args):
    z, membership = mixed_cluster_check(data, '铅钡')
    fig, ax = plt.subplots(figsize=(9, max(6, len(membership)*.19)), layout='constrained')
    dendrogram(z, labels=membership['采样点'].tolist(), orientation='right', leaf_font_size=7,
               color_threshold=0, above_threshold_color='#315b78', ax=ax)
    ax.set_xlabel('平均联接距离（闭合百分比）')
    ax.set_title('铅钡：原始与预测数据的混合聚类\n* 为修正版预测；不是预测准确性的独立验证')
    finish(fig, args, 'appendixA_lead_barium_mixed_cluster')


if __name__ == "__main__":
    figure_main(draw, __doc__)
