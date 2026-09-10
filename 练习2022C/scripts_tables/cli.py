"""全部解题/单图入口共用的参数与路径。"""
import argparse
from pathlib import Path
from .constants import DEFAULT_DATA, ROOT, SEED


# 创建各分析和绘图入口共用的命令行参数解析器
def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--data', type=Path, default=DEFAULT_DATA, help='原始附件.xlsx路径')
    p.add_argument('--output', type=Path, default=ROOT / 'outputs', help='结果根目录')
    p.add_argument('--seed', type=int, default=SEED)
    p.add_argument('--permutation-repeats', type=int, default=20000)
    p.add_argument('--sensitivity-repeats', type=int, default=200, help='每个扰动场景的重复次数')
    p.add_argument('--format', choices=['png', 'pdf', 'both'], default='png')
    p.add_argument('--dpi', type=int, default=180)
    return p

