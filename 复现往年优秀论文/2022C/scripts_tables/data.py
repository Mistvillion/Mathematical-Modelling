"""从原始三个表单读取，区分文物、采样点、作者推定风化组。"""
from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np
import pandas as pd
from .constants import COMPONENTS, DEFAULT_DATA, PAPER_STAGES
from .composition import closure, paper_clr, replaced_clr


@dataclass
class Data:
    artifacts: pd.DataFrame
    samples: pd.DataFrame
    invalid: pd.DataFrame
    unknown: pd.DataFrame
    path: Path

    # 提取样本成分矩阵并转换到指定数据空间
    def matrix(self, frame=None, space='raw'):
        frame = self.samples if frame is None else frame
        x = frame[COMPONENTS].to_numpy(float)
        if space == 'raw':
            return x
        if space == 'closed':
            return closure(x)
        if space == 'paper_clr':
            return paper_clr(x)[0]
        if space == 'replaced_clr':
            return replaced_clr(x)
        raise ValueError(f'未知数据空间: {space}')


# 整理化学成分列并校验含量与合计的有效性
def _composition_columns(frame, start):
    found = [re.search(r'\(([^)]+)\)', str(c)) for c in frame.columns[start:]]
    if any(m is None for m in found) or [m.group(1) for m in found] != COMPONENTS:
        raise ValueError('化学成分列名/顺序与2022C原附件不一致。')
    frame = frame.rename(columns=dict(zip(frame.columns[start:], COMPONENTS)))
    frame[COMPONENTS] = frame[COMPONENTS].apply(pd.to_numeric, errors='raise').fillna(0.0)
    x = frame[COMPONENTS].to_numpy(float)
    if not np.isfinite(x).all() or (x < 0).any():
        raise ValueError('原始含量存在负数或非有限值。')
    frame['含量合计'] = x.sum(axis=1)
    frame['有效'] = frame['含量合计'].between(85, 105, inclusive='both')
    return frame


# 读取附件三个表单并整理有效样本、风化标签和未知样本
def load_data(path=None):
    path = Path(path or DEFAULT_DATA).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'找不到附件: {path}；请用 --data 指定原始附件.xlsx。')
    sheets = pd.read_excel(path, sheet_name=['表单1', '表单2', '表单3'], dtype={'文物编号': str, '文物采样点': str})
    artifacts = sheets['表单1'].copy()
    artifacts['文物编号'] = artifacts['文物编号'].str.zfill(2)
    if artifacts['文物编号'].duplicated().any():
        raise ValueError('表单1的文物编号必须唯一。')
    if not artifacts['类型'].isin(['高钾', '铅钡']).all():
        raise ValueError('未知玻璃类型。')
    samples = _composition_columns(sheets['表单2'], 1)
    samples['文物编号'] = samples['文物采样点'].str.extract(r'^(\d+)')[0].str.zfill(2)
    if samples['文物采样点'].duplicated().any():
        raise ValueError('采样点名称必须唯一。')
    samples = samples.merge(artifacts, on='文物编号', how='left', validate='many_to_one', indicator=True)
    if not samples['_merge'].eq('both').all():
        raise ValueError('存在无法与表单1匹配的采样点。')
    samples = samples.drop(columns='_merge').rename(columns={'表面风化': '文物风化'})
    samples['点位风化'] = samples['文物风化']
    samples.loc[samples['文物采样点'].str.contains('未风化点'), '点位风化'] = '无风化'
    samples.loc[samples['文物采样点'].str.contains('严重风化点'), '点位风化'] = '风化'
    stage_map = {name: stage for groups in PAPER_STAGES.values() for stage, names in groups.items() for name in names}
    samples['论文阶段'] = samples['文物采样点'].map(stage_map)
    samples['论文风化'] = samples['论文阶段'].map(lambda s: '风化' if pd.notna(s) and s >= 3 else '无风化')
    invalid = samples.loc[~samples['有效']].copy().reset_index(drop=True)
    samples = samples.loc[samples['有效']].copy().reset_index(drop=True)
    if samples['论文阶段'].isna().any():
        raise ValueError('存在不在论文表11/12中的有效采样点，不能套用固定阶段。')
    samples['论文阶段'] = samples['论文阶段'].astype(int)
    unknown = _composition_columns(sheets['表单3'], 2)
    if not unknown['表面风化'].isin(['无风化', '风化']).all():
        raise ValueError('表单3风化标签不受支持。')
    return Data(artifacts, samples, invalid, unknown, path)


# 将结果表保存到输出目录下的CSV文件
def save_table(frame, output, name, index=False):
    path = Path(output) / 'tables' / f'{name}.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=index, encoding='utf-8-sig', float_format='%.10g')
    return path


# 导出原始数据、剔除记录、成分变换和非零掩码
def preprocess(data, output):
    save_table(data.artifacts, output, '00_文物信息')
    save_table(data.samples, output, '00_有效采样点_原始含量')
    save_table(data.invalid, output, '00_剔除记录')
    save_table(data.unknown, output, '00_未知样本_原始含量')
    meta = data.samples.drop(columns=COMPONENTS)
    for space in ['closed', 'paper_clr', 'replaced_clr']:
        transformed = pd.DataFrame(data.matrix(space=space), columns=COMPONENTS)
        save_table(pd.concat([meta, transformed], axis=1), output, f'00_{space}')
    mask = pd.DataFrame(data.matrix() > 0, columns=COMPONENTS)
    save_table(pd.concat([data.samples[['文物采样点']], mask], axis=1), output, '00_非零掩码')
