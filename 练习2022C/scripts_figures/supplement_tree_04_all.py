"""补充 无风化14成分自由选择重训树。来源：对应PDF p18-20。新增训练结果，非论文原树。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scripts_tables.plotting import plt, finish, figure_main
from sklearn.tree import plot_tree
from scripts_tables.classification import fitted_models


# 绘制并保存高钾玻璃无风化组的14成分重训决策树
def draw(data, args):
    model = fitted_models(data, args.seed)['无风化']
    fig, ax = plt.subplots(figsize=(8, 5.2), layout='constrained')
    plot_tree(model, feature_names=['SiO2', 'Na2O', 'K2O', 'CaO', 'MgO', 'Al2O3', 'Fe2O3', 'CuO', 'PbO', 'BaO', 'P2O5', 'SrO', 'SnO2', 'SO2'], class_names=model.classes_.tolist(),
              filled=True, rounded=True, precision=3, fontsize=10, ax=ax)
    ax.set_title('无风化：14成分自由选择重训树（全部有效样本）')
    finish(fig, args, 'supplement_tree_04_all')


if __name__ == "__main__":
    figure_main(draw, __doc__)
