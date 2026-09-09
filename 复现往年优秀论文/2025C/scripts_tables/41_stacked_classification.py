"""C132：classification独立计算入口。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts_tables.pipeline import step_main

if __name__ == '__main__':
    step_main('classification')
