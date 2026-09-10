"""图1d 高钾无风化箱线图。来源：PDF p11。按正文补写；文物风化分组，论文非零子组成log-ratio。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scripts_tables.plotting import plt, finish, figure_main
from scripts_tables.constants import COMPONENTS
from scripts_tables.subclasses import subgroup


# 绘制并保存高钾无风化样本的成分对数比箱线图
def draw(data, args):
    frame = subgroup(data, '高钾', '无风化')
    x = data.matrix(frame, 'paper_clr')
    fig, ax = plt.subplots(figsize=(8.4, 6.1), layout='constrained')
    boxes = ax.boxplot(x, orientation='horizontal', tick_labels=COMPONENTS, patch_artist=True,
                      widths=.55, medianprops={'color': '#bd4936', 'linewidth': 1.5},
                      flierprops={'marker': '.', 'markersize': 4, 'markerfacecolor': '#bd4936'})
    for patch in boxes['boxes']:
        patch.set_facecolor('#b8d7e8')
    ax.axvline(0, color='#888888', lw=.7, ls='--')
    ax.grid(axis='x', alpha=.2)
    ax.set_xlabel('非零子组成中心化对数比（无量纲）')
    ax.set_title('高钾无风化：14种成分分布（n=' + str(len(frame)) + '）')
    finish(fig, args, 'fig01d_boxplot')


if __name__ == "__main__":
    figure_main(draw, __doc__)
