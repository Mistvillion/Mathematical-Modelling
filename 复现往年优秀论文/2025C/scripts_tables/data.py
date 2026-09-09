"""读取原始两张表；不按GC阈值删记录，不把出生结局当模型特征。"""
from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np
import pandas as pd
from .constants import COLUMNS, DATA, THRESHOLD


def parse_week(value):
    """16w+1 -> 16+1/7；天数只接受0..6，坏值明确报错。"""
    text = str(value).strip().lower()
    match = re.fullmatch(r'(\d+)\s*w(?:\s*\+\s*([0-6]))?', text)
    if match:
        result = int(match[1]) + int(match[2] or 0) / 7
        if not 0 < result <= 45:
            raise ValueError(f'孕周超出可解析范围: {value!r}')
        return result
    try:
        result = float(text)
    except ValueError as exc:
        raise ValueError(f'无法识别孕周: {value!r}') from exc
    if not np.isfinite(result) or not 0 < result <= 45:
        raise ValueError(f'孕周超出可解析范围: {value!r}')
    return result


@dataclass
class Data:
    male: pd.DataFrame
    female: pd.DataFrame
    audit: pd.DataFrame
    source: Path

    @property
    def combined(self):
        return pd.concat([self.male, self.female], ignore_index=True)


def load_data(path=DATA):
    path = Path(path).resolve()
    book = pd.read_excel(path, sheet_name=None)
    result, audits = {}, []
    for sex, sheet in [('男', '男胎检测数据'), ('女', '女胎检测数据')]:
        if sheet not in book:
            raise ValueError(f'附件缺少工作表: {sheet}')
        raw = book[sheet].copy()
        raw.columns = raw.columns.astype(str).str.strip()
        raw = raw.loc[:, ~raw.columns.str.startswith('Unnamed:')]
        for col in raw:
            if raw[col].isna().any():
                audits.append({'sex': sex, 'item': f'原始缺失:{col}', 'count': int(raw[col].isna().sum())})
        frame = raw.rename(columns=COLUMNS)
        required = ['row_id', 'person_id', 'age', 'height', 'weight', 'bmi', 'week_text',
                    'z13', 'z18', 'z21', 'zx', 'gc', 'abnormal_text'] + (['y'] if sex == '男' else [])
        missing = set(required) - set(frame)
        if missing:
            raise ValueError(f'{sheet}缺列: {sorted(missing)}')
        if frame['row_id'].duplicated().any() or frame['person_id'].isna().any():
            raise ValueError(f'{sheet}存在重复行号或缺失孕妇代码')
        frame['sex'] = sex
        frame['week'] = frame['week_text'].map(parse_week)
        for col in ['age', 'height', 'weight', 'bmi', 'reads', 'unique_reads', 'draw', 'gc',
                    'mapping', 'duplicate', 'filtered', 'z13', 'z18', 'z21', 'zx', 'gc13',
                    'gc18', 'gc21', 'x', 'parity'] + (['y', 'zy'] if sex == '男' else []):
            frame[col] = pd.to_numeric(frame[col], errors='raise')
        fill = frame.bmi.isna()
        frame['bmi_imputed'] = fill
        # BMI有确定的物理定义，避免全量拟合填补带来的泄漏。
        frame.loc[fill, 'bmi'] = frame.loc[fill, 'weight'] / (frame.loc[fill, 'height'] / 100) ** 2
        if frame[['week', 'bmi', 'age', 'height', 'weight']].isna().any().any():
            raise ValueError('核心生理字段仍有缺失，不能静默丢弃孕妇')
        if (frame[['bmi', 'height', 'weight']] <= 0).any().any():
            raise ValueError('BMI、身高、体重必须为正数')
        if sex == '男' and (frame.y.isna().any() or not frame.y.between(0, 1).all()):
            raise ValueError('Y浓度应以0..1比例记录，4%应为0.04')
        for chrom in (13, 18, 21):
            frame[f't{chrom}'] = frame.abnormal_text.fillna('').str.contains(f'T{chrom}', regex=False).astype(int)
        frame['state'] = frame.t13 + 2 * frame.t18 + 4 * frame.t21
        frame['abnormal'] = (frame.state > 0).astype(int)
        frame['interaction'] = (frame[['t13', 't18', 't21']].sum(axis=1) >= 2).astype(int)
        frame['birth_unhealthy'] = frame.health_text.map({'是': 0, '否': 1})
        frame['ivf'] = (~frame.ivf_text.astype(str).str.contains('自然')).astype(int)
        frame['pregnancies'] = pd.to_numeric(frame.pregnancies_text.astype(str).str.replace('≥', '', regex=False))
        frame['gc_in_range'] = frame.gc.between(0.4, 0.6)
        frame['unique_ratio'] = frame.unique_reads / frame.reads
        frame = frame.sort_values(['person_id', 'week', 'row_id']).reset_index(drop=True)
        for item, count in [('记录数', len(frame)), ('孕妇数', frame.person_id.nunique()),
                            ('BMI物理公式填补', fill.sum()), ('GC在40%-60%内', frame.gc_in_range.sum()),
                            ('AB非整倍体记录', frame.abnormal.sum()),
                            ('同一孕妇同孕周额外记录', frame.duplicated(['person_id', 'week']).sum())]:
            audits.append({'sex': sex, 'item': item, 'count': int(count)})
        result[sex] = frame
    return Data(result['男'], result['女'], pd.DataFrame(audits), path)


def subjects(male):
    """同时记录最早实测达标、左/区间/右删失，绝不把未达标写成40周达标。"""
    rows = []
    for pid, group in male.groupby('person_id', sort=True):
        group = group.sort_values(['week', 'row_id'])
        first = group.iloc[0]
        hits = group.loc[group.y >= THRESHOLD, 'week']
        event = not hits.empty
        upper = float(hits.min()) if event else np.inf
        prior = group.loc[(group.week < upper) & (group.y < THRESHOLD), 'week']
        lower = float(prior.max()) if len(prior) else 0.
        censor = '区间' if event and lower else '左' if event else '右'
        rows.append({'person_id': pid, 'age': first.age, 'height': first.height,
                     'weight': first.weight, 'bmi': first.bmi, 'mean_bmi': group.bmi.mean(),
                     'n_records': len(group), 'n_weeks': group.week.nunique(),
                     'first_week': first.week, 'last_week': group.week.max(),
                     'first_observed_hit': upper, 'interval_lower': lower,
                     'interval_upper': upper, 'observed_event': int(event), 'censor_type': censor})
    return pd.DataFrame(rows)


def preprocess(data):
    numeric = ['age', 'height', 'weight', 'bmi', 'week', 'gc', 'z13', 'z18', 'z21', 'zx']
    overview = data.combined.groupby('sex')[numeric].agg(['count', 'mean', 'std', 'min', 'median', 'max'])
    overview.columns = [f'{a}_{b}' for a, b in overview.columns]
    return {'00_数据审计': data.audit, '00_男胎清洗数据': data.male,
            '00_女胎清洗数据': data.female, '00_男胎孕妇汇总': subjects(data.male),
            '00_描述统计': overview.reset_index(),
            '00_女胎标签分布': data.female.groupby('state').agg(
                records=('row_id', 'size'), subjects=('person_id', 'nunique')).reset_index()}
