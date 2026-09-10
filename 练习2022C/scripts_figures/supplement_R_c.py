"""补充 铅钡无风化R型变量树。来源：对应PDF p20-22。新增：按公式13/14重算R型聚类树；恒定列排除。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scripts_tables.plotting import plt, finish, figure_main
from scipy.cluster.hierarchy import dendrogram
from scripts_tables.subclasses import subgroup, r_cluster


# 绘制并保存铅钡无风化样本各非恒定成分的R型变量聚类树
def draw(data, args):
    frame = subgroup(data, '铅钡', '无风化')
    z, names, _, _ = r_cluster(data.matrix(frame))
    fig, ax = plt.subplots(figsize=(8, 5.4), layout='constrained')
    dendrogram(z, labels=names, orientation='right', leaf_font_size=10,
               color_threshold=0, above_threshold_color='#315b78', ax=ax)
    ax.set_title('铅钡无风化：R型变量聚类（原始含量）')
    ax.set_xlabel('single linkage，变量距离 1-|r|')
    ax.grid(axis='x', alpha=.2)
    finish(fig, args, 'supplement_R_c')


if __name__ == "__main__":
    figure_main(draw, __doc__)
