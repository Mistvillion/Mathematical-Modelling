"""问题1.2：四组含量与log-ratio描述统计。根据C155正文补写，不是作者原程序。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts_tables.cli import parser
from scripts_tables.data import load_data
from scripts_tables.descriptive import analyze


# 读取命令行参数和附件并生成成分描述统计表
def main():
    args = parser(__doc__).parse_args()
    data = load_data(args.data)
    analyze(data, args.output)
    print(f'结果已保存至 {args.output / "tables"}')


if __name__ == '__main__':
    main()
