"""补充 风化按原文预选PbO重训树。来源：对应PDF p18-20。新增训练结果，非论文原树。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scripts_tables.plotting import plt, finish, figure_main
from sklearn.tree import plot_tree
from scripts_tables.classification import fitted_pbo_models


# 绘制并保存高钾玻璃风化组的PbO重训决策树
def draw(data, args):
    model = fitted_pbo_models(data, args.seed)['风化']
    fig, ax = plt.subplots(figsize=(8, 5.2), layout='constrained')
    plot_tree(model, feature_names=['PbO'], class_names=model.classes_.tolist(),
              filled=True, rounded=True, precision=3, fontsize=10, ax=ax)
    ax.set_title('风化：按原文预选PbO重训树（全部有效样本）')
    finish(fig, args, 'supplement_tree_05_pbo')


if __name__ == "__main__":
    figure_main(draw, __doc__)
