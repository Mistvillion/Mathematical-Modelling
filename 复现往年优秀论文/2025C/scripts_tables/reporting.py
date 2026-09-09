"""CSV、Excel和每步来源记录；不跨参数静默复用旧结果。"""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import pandas as pd
from .constants import PROJECT, PAPER


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def signature(args):
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
              if key not in ['output', 'dpi', 'format', 'tables_only', 'resume']}
    numerical = ['constants.py', 'data.py', 'mixed.py', 'gpr.py', 'optimization.py', 'survival.py', 'classification.py', 'pipeline.py']
    return {'data_sha256': sha256(args.data), 'config': config,
            'calculation_sha256': {name: sha256(PROJECT / 'scripts_tables' / name) for name in numerical}}


def save_tables(tables, args, step):
    folder = args.output / 'tables'
    folder.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(folder / f'{name}.csv', index=False, encoding='utf-8-sig', float_format='%.12g')
    path = args.output / 'run_metadata.json'
    meta = json.loads(path.read_text()) if path.exists() else {'steps': {}}
    meta['steps'][step] = {**signature(args), 'tables': list(tables),
                           'table_sha256': {name: sha256(folder / f'{name}.csv') for name in tables},
                           'completed_utc': datetime.now(timezone.utc).isoformat()}
    meta.update({'python': platform.python_version(), 'platform': platform.platform(),
                 'paper_sha256': sha256(PAPER), 'paper': str(PAPER),
                 'packages': {name: importlib.metadata.version(name) for name in [
                     'numpy', 'pandas', 'scipy', 'statsmodels', 'scikit-learn', 'matplotlib',
                     'openpyxl', 'torch', 'lightgbm', 'optuna']},
                 'source_sha256': {str(p.relative_to(PROJECT)): sha256(p) for p in sorted(PROJECT.glob('scripts_*/*.py'))}})
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n')
    print(f'  {step}：保存{len(tables)}张表', flush=True)


def cached(name, args, step):
    path = args.output / 'run_metadata.json'
    if not path.exists():
        return None
    meta = json.loads(path.read_text())
    entry = meta.get('steps', {}).get(step, {})
    if any(entry.get(key) != value for key, value in signature(args).items()):
        return None
    csv = args.output / 'tables' / f'{name}.csv'
    if not csv.exists() or entry.get('table_sha256', {}).get(name) != sha256(csv):
        return None
    return pd.read_csv(csv)


def summarize(args):
    meta_path = args.output / 'run_metadata.json'
    meta = json.loads(meta_path.read_text())
    meta['source_sha256'] = {str(p.relative_to(PROJECT)): sha256(p) for p in sorted(PROJECT.glob('scripts_*/*.py'))}
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n')
    active = []
    for step, entry in meta['steps'].items():
        for name in entry['tables']:
            active.append((step, name, entry['data_sha256']))
    index = pd.DataFrame(active, columns=['step', 'table', 'data_sha256'])
    with pd.ExcelWriter(args.output / '计算结果汇总.xlsx', engine='openpyxl') as writer:
        index.to_excel(writer, sheet_name='目录', index=False)
        used = {'目录'}
        for _, name, _ in active:
            label = name[:31]
            counter = 1
            while label in used:
                label = name[:27] + f'_{counter}'
                counter += 1
            used.add(label)
            pd.read_csv(args.output / 'tables' / f'{name}.csv').to_excel(writer, sheet_name=label, index=False, inf_rep='∞')
        for sheet in writer.book:
            sheet.freeze_panes = 'A2'
            sheet.auto_filter.ref = sheet.dimensions
            for col in sheet.columns:
                letter = col[0].column_letter
                sheet.column_dimensions[letter].width = min(36, max(12, max(len(str(c.value or '')) for c in list(col)[:100]) + 2))
    text = ['# 运行结果', '', '以下数值由当前结果CSV自动生成；原论文对照值不参与计算。', '']
    snippets = [('13_模型检验', '问题一'), ('22_GA与动态规划对照', '问题二：GA全局风险审计'),
                ('31_生存模型评估', '问题三：训练、验证与测试'), ('42_总体与分折指标', '问题四：按孕妇分组验证')]
    for name, title in snippets:
        path = args.output / 'tables' / f'{name}.csv'
        if path.exists():
            frame = pd.read_csv(path)
            if name.startswith('22'):
                frame = frame[frame.k == 3]
            if name.startswith('42'):
                frame = frame[frame.scope == 'OOF_all']
            text.extend([f'## {title}', '', frame.to_markdown(index=False, floatfmt='.5g'), ''])
    text.extend(['## 解释范围', '',
                 '- 问题二的达标比例是组内经验分布比例，不等于NIPT诊断准确率。',
                 '- paper_history使用完整轨迹和GPR伪标签，属于回顾性重构；static_interval是补充的基线区间删失对照。',
                 '- 问题四的OOF预测来自外层按孕妇分组的未参与该折训练的数据；不宣称复现原文99.7%。',
                 '- 详细差异、公式和数据口径见 ../docs/复现说明.md。', ''])
    (args.output / '运行结果.md').write_text('\n'.join(text))
