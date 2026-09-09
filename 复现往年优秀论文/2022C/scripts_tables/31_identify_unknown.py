"""问题3：未知样本分类与敏感性。根据C155正文补写，不是作者原程序。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts_tables.cli import parser
from scripts_tables.data import load_data
from scripts_tables.classification import analyze_unknown


# 读取命令行参数和附件并执行未知样本分类与敏感性分析
def main():
    args = parser(__doc__).parse_args()
    data = load_data(args.data)
    analyze_unknown(data, args.output, args.sensitivity_repeats, args.seed)
    print(f'结果已保存至 {args.output / "tables"}')


if __name__ == '__main__':
    main()
