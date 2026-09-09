"""校验已有结果，生成outputs/验证记录.md和validation.json。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts_tables.cli import parse
from scripts_tables.validation import validate

if __name__ == '__main__':
    validate(parse('C132：验证已有结果'))
