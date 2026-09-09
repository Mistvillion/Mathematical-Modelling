"""结果打包与运行清单；所有统计值来自本次计算。"""
from pathlib import Path
from os.path import relpath
from urllib.parse import quote
import hashlib
import importlib.metadata
import json
import pandas as pd
from .constants import COMPONENTS, ROOT


# 将CSV结果表汇总为带索引的Excel工作簿
def export_workbook(output):
    """CSV保留原有细分文件；Excel汇总方便直接查表，首张sheet提供完整文件名。"""
    output = Path(output)
    paths = sorted((output / 'tables').glob('*.csv'))
    with pd.ExcelWriter(output / '计算结果汇总.xlsx', engine='openpyxl') as writer:
        index = []
        for i, path in enumerate(paths, 1):
            sheet = f'{i:02d}_{path.stem}'[:31]
            frame = pd.read_csv(path, dtype={'文物编号': str, '文物采样点': str, '采样点': str})
            frame.to_excel(writer, sheet_name=sheet, index=False)
            index.append({'工作表': sheet, 'CSV文件': path.name, '行数': len(frame)})
        pd.DataFrame(index).to_excel(writer, sheet_name='索引', index=False)
        writer.book.active = len(index)


# 保存运行参数、依赖版本和数据摘要并在结果齐备时生成报告
def save_run_info(data, args, figures):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    versions = {name: importlib.metadata.version(name) for name in
                ['numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib', 'openpyxl']}
    meta = {'附件sha256': hashlib.sha256(data.path.read_bytes()).hexdigest(),
            'seed': args.seed, 'permutation_repeats': args.permutation_repeats,
            'sensitivity_repeats': args.sensitivity_repeats,
            '文物数': len(data.artifacts), '有效采样点': len(data.samples),
            '各类型采样点': data.samples['类型'].value_counts().to_dict(),
            '剔除采样点': data.invalid['文物采样点'].tolist(), '未知样本数': len(data.unknown),
            '本次图形脚本数': figures, '依赖版本': versions,
            '回归中心顺序': '表13/14；高钾阶段3/4与表11相反',
            '预测修正': '实际拟合阶段 + signed_sqrt残差 + 独立原始零mask',
            '亚类输入': '文物风化分组 + 原始含量 + 表25特征 + Ward k=2'}
    (output / 'run_metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n')
    if (output / 'tables/31_未知样本分类与边界距离.csv').exists():
        unknown = pd.read_csv(output / 'tables/31_未知样本分类与边界距离.csv')
        predicted = pd.read_csv(output / 'tables/13_风化前预测_signed_sqrt.csv')
        max_error = abs(predicted[COMPONENTS].sum(axis=1) - 100).max()
        docs = quote(Path(relpath(ROOT / 'docs', output.resolve())).as_posix())
        lines = ['# 本次运行结果', '',
                 f'文物 {len(data.artifacts)} 件；有效采样点 {len(data.samples)} 个；剔除：' + '、'.join(meta['剔除采样点']) + '。', '',
                 f'风化前预测 {len(predicted)} 行；CSV读回后的行和最大误差 {max_error:.3g} 个百分点。', '',
                 '| 文物 | 论文PbO规则 | 预选PbO重训树 | 全14成分重训树 | Q聚类对照 |', '|---|---|---|---|---|']
        for _, r in unknown.iterrows():
            lines.append('| ' + ' | '.join(str(r[c]) for c in ['文物编号', '论文固定规则', 'PbO重训树预测', '重训树预测', 'Q聚类对照']) + ' |')
        lines.extend(['', '“论文标签”用于方法复现对照，不是表单3外部检测真值。Q聚类也不是交叉验证。', '',
                      f'本次生成 {figures} 张图；每张图的来源与脚本见 [图表索引]({docs}/图表索引.md)。', '',
                      '回归表15/16可以用表13/14中心顺序核对；预测表17/18和灰色关联表28不宣称逐值复现。',
                      f'统计选择、阶段矛盾、模型适用范围详见 [复现说明]({docs}/复现说明.md)与[论文评价]({docs}/论文评价.md)。'])
        (output / '运行结果.md').write_text('\n'.join(lines) + '\n')
