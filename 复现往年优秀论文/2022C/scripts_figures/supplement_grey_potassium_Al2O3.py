"""补充 高钾Al2O3灰色关联。来源：PDF p25-28。正文声称遍历但PDF未附该母序列图；补齐计算图。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scripts_tables.plotting import plt, finish, figure_main
from scripts_tables.constants import COMPONENTS
from scripts_tables.grey import grey_coefficients


# 绘制并保存高钾玻璃以Al2O3为母序列的灰色关联系数曲线
def draw(data, args):
    frame = data.samples.loc[data.samples['类型'].eq('高钾')]
    coeff = grey_coefficients(data.matrix(frame), reference=5, rho=.5)
    fig, ax = plt.subplots(figsize=(12, 5.8), layout='constrained')
    colors = plt.get_cmap('tab20').colors
    for i, chemical in enumerate(COMPONENTS):
        if i == 5 or not np.isfinite(coeff[:, i]).any():
            continue
        ax.plot(range(len(frame)), coeff[:, i], label=chemical, lw=1.05, marker='.',
                markersize=3, color=colors[i])
    step = 1 if len(frame) < 25 else 2
    loc = np.arange(0, len(frame), step)
    ax.set_xticks(loc, frame['文物采样点'].iloc[loc].tolist(), rotation=65, ha='right', fontsize=7)
    ax.set_ylim(0, 1.04)
    ax.set_title('高钾玻璃：以Al2O3为母序列的灰色关联系数')
    ax.set_xlabel('采样点（附件顺序，不是时间）')
    ax.set_ylabel('关联系数（均值无量纲，ρ=0.5）')
    ax.grid(alpha=.2)
    ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.08), ncol=7, frameon=False, fontsize=8)
    finish(fig, args, 'supplement_grey_potassium_Al2O3')


if __name__ == "__main__":
    figure_main(draw, __doc__)
