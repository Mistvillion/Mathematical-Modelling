"""图7a 高钾无风化亚类成员。来源：PDF p23。按表25特征重算的成员图；不冒充原图逐成员结果。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scripts_tables.plotting import plt, finish, figure_main
from scripts_tables.subclasses import subclass_result


# 绘制并保存高钾无风化样本的亚类成员图
def draw(data, args):
    frame, features, x, labels = subclass_result(data, '高钾', '无风化')
    left = frame.loc[labels == 0, '文物采样点'].tolist()
    right = frame.loc[labels == 1, '文物采样点'].tolist()
    max_lines = max(len(left), len(right))
    fig, ax = plt.subplots(figsize=(9.5, max(5.4, 2.8+max_lines*.20)), layout='constrained')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.text(.5, .88, '高钾无风化', ha='center', bbox={'boxstyle':'round', 'fc':'#b8d7e8', 'ec':'#315b78'}, fontsize=14)
    for xpos, names, label in [(.25, left, 1), (.75, right, 2)]:
        ax.annotate('', (xpos, .73), (.5, .84), arrowprops={'arrowstyle':'->','color':'#315b78'})
        ax.text(xpos, .72, '亚类' + str(label) + '（n=' + str(len(names)) + '）\n\n' + '\n'.join(names),
                ha='center', va='top', fontsize=9,
                bbox={'boxstyle':'round,pad=.7','fc':'#f1f5f7','ec':'#7c9cad'})
    ax.set_title('亚类划分：高钾无风化\n特征：' + ', '.join(features) + '；原始含量，Ward，k=2')
    finish(fig, args, 'fig07a_subclass_members')


if __name__ == "__main__":
    figure_main(draw, __doc__)
