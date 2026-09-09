"""问题1第一小问：文物级列联检验。稀疏RxC表用固定边缘Monte Carlo补充。"""
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact, random_table
from .constants import SEED
from .data import save_table


# 检验列联表独立性并根据稀疏程度选择推荐检验方法
def contingency_test(table, repeats=20000, seed=SEED):
    observed = np.asarray(table, dtype=int)
    if repeats < 1:
        raise ValueError('Monte Carlo重复次数须为正。')
    stat, p, dof, expected = chi2_contingency(observed, correction=False)
    sparse = (expected < 1).any() or (expected < 5).mean() > 0.2
    method, recommended, se = 'Pearson', p, np.nan
    if sparse and observed.shape == (2, 2):
        method, recommended = 'Fisher双侧精确', fisher_exact(observed).pvalue
    elif sparse:
        draws = random_table.rvs(observed.sum(1), observed.sum(0), size=repeats,
                                 random_state=np.random.default_rng(seed))
        simulated = ((draws - expected) ** 2 / expected).sum(axis=(1, 2))
        recommended = (np.count_nonzero(simulated >= stat - 1e-12) + 1) / (repeats + 1)
        se = np.sqrt(recommended * (1 - recommended) / repeats)
        method = '固定边缘MonteCarlo-Pearson统计量'
    result = {'n': int(observed.sum()), 'chi2': stat, 'df': dof, 'Pearson_p': p,
              '期望小于5比例': (expected < 5).mean(), '最小期望': expected.min(),
              '推荐方法': method, '推荐p': recommended, 'MonteCarlo标准误': se,
              '重复次数': repeats if 'MonteCarlo' in method else 0, 'seed': seed,
              'CramersV': np.sqrt(stat / (observed.sum() * min(np.array(observed.shape) - 1))),
              '结论_5%': '有统计关联证据' if recommended < 0.05 else '未发现充分关联证据'}
    return result, expected


# 分析文物风化与类型、纹饰和颜色的关联并保存结果
def analyze(data, output, repeats=20000, seed=SEED):
    rows = []
    for factor, ref_p in [('类型', 0.009), ('纹饰', 0.084), ('颜色', 0.507)]:
        frame = data.artifacts.dropna(subset=[factor, '表面风化'])
        table = pd.crosstab(frame['表面风化'], frame[factor])
        result, expected = contingency_test(table, repeats, seed)
        rows.append({'因素': factor, **result, '论文p_仅参考': ref_p})
        save_table(table, output, f'11_{factor}_频数', index=True)
        save_table(pd.DataFrame(expected, index=table.index, columns=table.columns),
                   output, f'11_{factor}_期望频数', index=True)
    result = pd.DataFrame(rows)
    save_table(result, output, '11_独立性检验')
    return result
