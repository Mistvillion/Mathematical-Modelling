"""仅共享绘图样式/保存和入口；每张图的绘制逻辑在scripts_figures/各文件内。"""
from pathlib import Path
import os
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from .cli import parser
from .data import load_data


# 配置中文字体和共用绘图样式
def setup_style():
    # 可手动 C155_FONT=/path/to/ChineseFont.ttf；否则检测常见系统中文字体。
    font_file = os.environ.get('C155_FONT')
    if font_file:
        font_manager.fontManager.addfont(font_file)
        family = font_manager.FontProperties(fname=font_file).get_name()
    else:
        candidates = ['PingFang SC', 'Heiti SC', 'Songti SC', 'Microsoft YaHei',
                      'Noto Sans CJK SC', 'Source Han Sans SC', 'SimHei', 'Arial Unicode MS']
        available = {f.name for f in font_manager.fontManager.ttflist}
        family = next((f for f in candidates if f in available), None)
        if family is None:
            warnings.warn('未找到中文字体；请安装Noto Sans CJK或设置C155_FONT后重新绘图。')
            family = 'DejaVu Sans'
    plt.rcParams.update({'font.family': [family, 'DejaVu Sans'], 'axes.unicode_minus': False,
                         'font.size': 10, 'axes.titlesize': 13, 'axes.labelsize': 10,
                         'figure.facecolor': 'white', 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    return family


# 按指定格式保存图形并关闭画布
def finish(fig, args, name):
    folder = Path(args.output) / 'figures'
    folder.mkdir(parents=True, exist_ok=True)
    formats = ['png', 'pdf'] if args.format == 'both' else [args.format]
    for fmt in formats:
        fig.savefig(folder / f'{name}.{fmt}', dpi=args.dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# 解析绘图参数、配置样式并读取数据调用绘图函数
def figure_main(draw, description):
    args = parser(description).parse_args()
    setup_style()
    draw(load_data(args.data), args)
    print(f'图已保存至 {args.output / "figures"}')
