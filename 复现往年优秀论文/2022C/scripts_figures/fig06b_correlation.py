"""图6b 铅钡无风化相关矩阵。来源：PDF p21。按正文补写；非零子组成log-ratio；常量相关遮罩。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scripts_tables.plotting import plt, finish, figure_main
from scripts_tables.constants import COMPONENTS
from scripts_tables.subclasses import subgroup


# 绘制并保存铅钡无风化样本的成分对数比相关矩阵热图
def draw(data, args):
    frame = subgroup(data, '铅钡', '无风化')
    corr = pd.DataFrame(data.matrix(frame, 'paper_clr'), columns=COMPONENTS).corr()
    fig, ax = plt.subplots(figsize=(8.1, 7.0), layout='constrained')
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad('#b9b9b9')
    image = ax.imshow(np.ma.masked_invalid(corr), vmin=-1, vmax=1, cmap=cmap)
    ax.set_xticks(range(14), COMPONENTS, rotation=55, ha='right')
    ax.set_yticks(range(14), COMPONENTS)
    ax.set_title('铅钡无风化：log-ratio Pearson相关\n灰色为恒定列，相关系数未定义')
    fig.colorbar(image, ax=ax, shrink=.78, label='Pearson r（不是因果关系）')
    finish(fig, args, 'fig06b_correlation')


if __name__ == "__main__":
    figure_main(draw, __doc__)
