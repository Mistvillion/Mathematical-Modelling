"""命令行配置；缩减搜索预算与论文预算均显式可见。"""
import argparse
from pathlib import Path
from .constants import DATA, OUTPUT


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('必须为正整数')
    return value


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--data', type=Path, default=DATA)
    p.add_argument('--output', type=Path, default=OUTPUT)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--gpr-restarts', type=int, default=10)
    p.add_argument('--population', type=positive, default=100)
    p.add_argument('--generations', type=positive, default=50)
    p.add_argument('--epochs', type=positive, default=200)
    p.add_argument('--trials', type=positive, default=12, help='每个专家Optuna次数；原附录50')
    p.add_argument('--meta-trials', type=positive, default=20, help='元模型Optuna次数；原附录100')
    p.add_argument('--max-estimators', type=positive, default=500, help='LightGBM树数上限；原附录1000')
    p.add_argument('--folds', type=positive, default=3, help='按孕妇分组的外层验证折数')
    p.add_argument('--inner-folds', type=positive, default=3)
    p.add_argument('--sensitivity-repeats', type=positive, default=100)
    p.add_argument('--targets', type=int, nargs='+', default=[50, 75, 90, 95, 99])
    p.add_argument('--dense', action='store_true', help='达标比例遍历50..99，对应原附录曲线')
    p.add_argument('--tables-only', action='store_true', help='只计算表格，暂不绘图')
    p.add_argument('--resume', action='store_true', help='跳过输入、参数和计算源代码完全一致的已完成步骤')
    p.add_argument('--dpi', type=positive, default=180)
    p.add_argument('--format', choices=['png', 'pdf', 'both'], default='png')
    return p


def parse(description):
    args = parser(description).parse_args()
    args.data, args.output = args.data.resolve(), args.output.resolve()
    if args.gpr_restarts < 0:
        raise ValueError('GPR重启次数不能为负')
    if args.folds < 2 or args.inner_folds < 2:
        raise ValueError('交叉验证至少需要2折')
    args.targets = list(range(50, 100)) if args.dense else sorted(set(args.targets))
    if not args.targets or any(not 0 < p < 100 for p in args.targets):
        raise ValueError('目标达标比例必须严格介于0和100之间')
    if args.population < 4:
        raise ValueError('GA种群至少为4')
    return args
