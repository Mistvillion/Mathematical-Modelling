"""图4 无风化论文决策树。来源：PDF p19。原文阈值重绘；非重新训练结果。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scripts_tables.plotting import plt, finish, figure_main


# 绘制并保存论文无风化样本的PbO固定阈值决策树
def draw(data, args):
    fig, ax = plt.subplots(figsize=(7, 4.6), layout='constrained')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    box = dict(boxstyle='round,pad=.6', fc='#d8e8ee', ec='#315b78')
    ax.text(.5, .78, 'PbO 含量', ha='center', va='center', bbox=box, fontsize=14)
    ax.text(.23, .22, '高钾玻璃', ha='center', va='center', bbox=box, fontsize=13)
    ax.text(.77, .22, '铅钡玻璃', ha='center', va='center', bbox=box, fontsize=13)
    for x, label in [(.23, '≤ 8.495%'), (.77, '> 8.495%')]:
        ax.annotate('', (x, .31), (.5, .70), arrowprops={'arrowstyle': '->', 'color': '#315b78'})
        ax.text((x+.5)/2, .5, label, ha='center', color='#a54734', fontsize=12,
                bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 2})
    ax.set_title('无风化：论文给出的固定分类规则')
    ax.text(.5, .02, '阈值应用于原始百分含量；来源：C155 p18-19', ha='center', fontsize=9)
    finish(fig, args, 'fig04_paper_decision_tree')


if __name__ == "__main__":
    figure_main(draw, __doc__)
