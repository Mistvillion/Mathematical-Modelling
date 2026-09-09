"""附录B 铅钡第1分支二次聚类。来源：PDF p33-34。附录代码会生成的图，正文未单列；补写。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scripts_tables.plotting import plt, finish, figure_main
from scipy.cluster.hierarchy import linkage, dendrogram
from scripts_tables.weathering import ward_labels


# 绘制并保存铅钡样本初次二分后第1分支的Ward聚类树
def draw(data, args):
    frame = data.samples.loc[data.samples['类型'].eq('铅钡')]
    x = data.matrix(frame)
    outer = ward_labels(x)
    frame = frame.iloc[np.flatnonzero(outer == 0)]
    z = linkage(data.matrix(frame), method='ward')
    fig, ax = plt.subplots(figsize=(8, max(4.5, len(frame)*.23)), layout='constrained')
    dendrogram(z, labels=frame['文物采样点'].tolist(), orientation='right', leaf_font_size=8,
               color_threshold=0, above_threshold_color='#315b78', ax=ax)
    ax.set_xlabel('Ward 合并距离（原始百分比）')
    ax.set_title('铅钡：二类聚类后第1分支再次聚类\n分支编号由算法生成，没有时间含义')
    finish(fig, args, 'appendixB_lead_barium_branch1')


if __name__ == "__main__":
    figure_main(draw, __doc__)
