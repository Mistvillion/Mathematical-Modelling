"""问题4：均值无量纲后的灰色关联（p25-28）；这是形状相似度，不是相关系数。"""
import numpy as np
import pandas as pd
from .constants import COMPONENTS, TYPES, PAPER_GREY_SIO2
from .data import save_table


# 计算各成分相对指定母序列的逐采样点灰色关联系数
def grey_coefficients(x, reference=0, rho=.5):
    """对每个母序列分别计算全比较集的极小/极大差，等权平均采样点。

    排除自身比较及全零列；全零列均值为0，不能做均值无量纲。
    无量纲用原始非负含量，不对有正有负的CLR均值作除法。
    """
    x = np.asarray(x, float)
    if x.ndim != 2 or not np.isfinite(x).all() or (x < 0).any() or not 0 < rho <= 1:
        raise ValueError('灰色分析需要非负有限二维含量，rho须在(0,1]。')
    means = x.mean(axis=0)
    valid = means > 0
    coeff = np.full_like(x, np.nan)
    if not valid[reference]:
        return coeff
    z = np.divide(x, means, out=np.zeros_like(x), where=valid)
    comparisons = valid.copy()
    comparisons[reference] = False
    if comparisons.any():
        differences = abs(z[:, comparisons] - z[:, [reference]])
        d_min, d_max = differences.min(), differences.max()
        values = np.ones_like(differences) if d_max == 0 else (d_min + rho * d_max) / (differences + rho * d_max)
        coeff[:, comparisons] = values
    coeff[:, reference] = 1.0
    return coeff


# 依次以各成分为母序列计算平均灰色关联度矩阵
def grey_matrix(x, rho=.5):
    result = np.full((x.shape[1], x.shape[1]), np.nan)
    for ref in range(x.shape[1]):
        coeff = grey_coefficients(x, ref, rho)
        valid = np.isfinite(coeff).all(axis=0)
        result[ref, valid] = coeff[:, valid].mean(axis=0)
    return result


# 计算两类玻璃的灰色关联及其差异并保存论文对照结果
def analyze(data, output):
    matrices = {}
    curves, exclusions = [], []
    for glass in TYPES:
        frame = data.samples.loc[data.samples['类型'].eq(glass)]
        x = data.matrix(frame)
        matrix = grey_matrix(x)
        matrices[glass] = matrix
        save_table(pd.DataFrame(matrix, index=COMPONENTS, columns=COMPONENTS), output,
                   f'41_{glass}_灰色关联矩阵', index=True)
        for chemical in np.asarray(COMPONENTS)[x.mean(axis=0) == 0]:
            exclusions.append({'类型': glass, '成分': chemical, '原因': '均值为0无法无量纲'})
        for ref, reference in enumerate(COMPONENTS):
            coeff = grey_coefficients(x, ref)
            for j, comparison in enumerate(COMPONENTS):
                if ref == j:
                    continue
                for i, sample in enumerate(frame['文物采样点']):
                    curves.append({'类型': glass, '母序列': reference, '子序列': comparison,
                                   '采样点': sample, '关联系数': coeff[i, j]})
    save_table(pd.DataFrame(curves), output, '41_逐点灰色关联系数_全部母序列')
    save_table(pd.DataFrame(exclusions, columns=['类型', '成分', '原因']), output, '41_零均值成分排除记录')
    difference = matrices['高钾'] - matrices['铅钡']
    save_table(pd.DataFrame(difference, index=COMPONENTS, columns=COMPONENTS), output,
               '42_关联矩阵差异_高钾减铅钡', index=True)
    comparisons = []
    for glass in TYPES:
        for j, chemical in enumerate(COMPONENTS[1:]):
            computed, reference = matrices[glass][0, j + 1], PAPER_GREY_SIO2[glass][j]
            comparisons.append({'类型': glass, '母序列': 'SiO2', '子序列': chemical,
                                '重算关联度': computed, '论文表28_参考': reference,
                                '差值': computed - reference})
    save_table(pd.DataFrame(comparisons), output, '42_论文表28对照')
    rows = []
    for glass, matrix in matrices.items():
        for ref, name in enumerate(COMPONENTS):
            a = matrix[ref].copy()
            a[ref] = np.nan
            valid = np.flatnonzero(np.isfinite(a))
            rows.append({'类型': glass, '母序列': name, '有效比较数': len(valid),
                         '均值': np.nanmean(a), '标准差': np.nanstd(a),
                         '最大关联成分': COMPONENTS[valid[np.argmax(a[valid])]] if len(valid) else '',
                         '最小关联成分': COMPONENTS[valid[np.argmin(a[valid])]] if len(valid) else ''})
    save_table(pd.DataFrame(rows), output, '42_各母序列关联差异摘要')
    return matrices
