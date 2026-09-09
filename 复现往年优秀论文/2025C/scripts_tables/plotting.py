"""只共享样式、来源校验和保存；各张图的布局位于独立脚本。"""
import json
import os
from pathlib import Path
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
import pandas as pd
from .cli import parse
from .reporting import sha256

COLORS = ['#2878A5', '#DB7952', '#359C82', '#9270AC', '#C4A13B']


def setup_style():
    font_file = os.environ.get('C132_FONT')
    if font_file:
        font_manager.fontManager.addfont(font_file)
        family = font_manager.FontProperties(fname=font_file).get_name()
    else:
        candidates = ['PingFang SC', 'Heiti SC', 'Songti SC', 'Microsoft YaHei',
                      'Noto Sans CJK SC', 'Source Han Sans SC', 'SimHei', 'Arial Unicode MS']
        available = {f.name for f in font_manager.fontManager.ttflist}
        family = next((name for name in candidates if name in available), 'DejaVu Sans')
        if family == 'DejaVu Sans':
            warnings.warn('未找到中文字体；请设置C132_FONT为中文字体文件路径。')
    plt.rcParams.update({'font.family': [family, 'DejaVu Sans'], 'axes.unicode_minus': False,
                         'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 10,
                         'figure.facecolor': 'white', 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.prop_cycle': plt.cycler(color=COLORS),
                         'pdf.fonttype': 42})


class Tables:
    def __init__(self, args):
        self.args = args
        path = args.output / 'run_metadata.json'
        if not path.exists():
            raise FileNotFoundError('请先运行 scripts_tables/00_run_all.py --tables-only 生成结果表。')
        self.metadata = json.loads(path.read_text())
        self.data_hash = sha256(args.data)

    def __call__(self, name):
        entry = next((entry for entry in self.metadata['steps'].values() if name in entry['tables']), None)
        if entry is None or entry['data_sha256'] != self.data_hash:
            raise ValueError(f'{name}不存在或附件已变化，请重新运行对应计算入口。')
        path = self.args.output / 'tables' / f'{name}.csv'
        if not path.exists() or sha256(path) != entry['table_sha256'][name]:
            raise ValueError(f'{name}的CSV与运行记录不符，请重新计算。')
        return pd.read_csv(path)


def finish(fig, args, name):
    folder = args.output / 'figures'
    folder.mkdir(parents=True, exist_ok=True)
    for extension in ['png', 'pdf'] if args.format == 'both' else [args.format]:
        fig.savefig(folder / f'{name}.{extension}', dpi=args.dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def flow(ax, nodes, edges):
    """nodes=(x,y,w,h,text)，坐标0..1；edges是节点索引对。"""
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis('off')
    for start, end in edges:
        a, b = nodes[start], nodes[end]
        ax.annotate('', xy=(b[0], b[1] + b[3] / 2 + .018), xytext=(a[0], a[1] - a[3] / 2 - .018),
                    arrowprops={'arrowstyle': '->', 'color': '#758291', 'lw': 1.4, 'shrinkA': 0, 'shrinkB': 0}, zorder=1)
    for x, y, width, height, label in nodes:
        box = FancyBboxPatch((x - width / 2, y - height / 2), width, height,
                             boxstyle='round,pad=0.015', facecolor='#EDF5F9', edgecolor=COLORS[0], lw=1.2, zorder=2)
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center', fontsize=10, zorder=3)


def figure_main(draw, name, description):
    args = parse(description)
    setup_style()
    fig = draw(Tables(args), args)
    finish(fig, args, name)
    print(f'已绘制 {name}', flush=True)
