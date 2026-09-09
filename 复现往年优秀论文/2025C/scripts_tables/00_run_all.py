"""一键复算六个步骤、Excel、逐图脚本与结果验证。"""
import json
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts_tables.cli import parse
from scripts_tables.constants import PROJECT
from scripts_tables.data import load_data
from scripts_tables.pipeline import run_step
from scripts_tables.reporting import cached, summarize


def main():
    args = parse('2025C C132：一键复现')
    data = load_data(args.data)
    for step in ['preprocess', 'mixed', 'gpr', 'grouping', 'survival', 'classification']:
        path = args.output / 'run_metadata.json'
        previous = json.loads(path.read_text()).get('steps', {}).get(step, {}) if path.exists() else {}
        if args.resume and previous.get('tables') and all(cached(name, args, step) is not None for name in previous['tables']):
            print(f'复用已核验的步骤：{step}', flush=True)
        else:
            run_step(step, args, data)
    summarize(args)
    if not args.tables_only:
        for script in sorted((PROJECT / 'scripts_figures').glob('[0-9]*.py')):
            subprocess.run([sys.executable, str(script), '--output', str(args.output), '--data', str(args.data),
                            '--format', args.format, '--dpi', str(args.dpi)], check=True)
        from scripts_tables.validation import validate
        validate(args)
    print(f'已完成，结果目录：{args.output}', flush=True)


if __name__ == '__main__':
    main()
