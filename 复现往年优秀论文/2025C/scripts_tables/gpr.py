"""问题二：附录B.2的独立GPR及三阶段规则；未达标保留inf。"""
import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from .constants import THRESHOLD
from .data import subjects


def estimate(male, seed=42, restarts=10):
    weeks = np.linspace(10, 40, 301)
    result, curves, logs = [], [], []
    summary = subjects(male).set_index('person_id')
    for pid, group in male.groupby('person_id', sort=True):
        group = group.sort_values(['week', 'row_id'])
        x, y = group.week.to_numpy(), group.y.to_numpy()
        preliminary = np.inf
        corr = pearsonr(x, y).statistic if len(group) > 1 and np.ptp(x) > 0 and np.ptp(y) > 0 else np.nan
        method, fitted_kernel, warning_text = '无增长趋势', '', ''
        if len(group) == 1:
            method = '单点观测'
            if y[0] >= THRESHOLD:
                preliminary = x[0]
        elif np.isfinite(corr) and corr >= 0:
            kernel = ConstantKernel(1.) * RBF(1., (1e-2, 1e2)) + WhiteKernel(1e-5, (1e-10, 1e1))
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=restarts,
                                          random_state=seed, normalize_y=False, alpha=1e-10)
            # 优化器的边界提示完整写入日志；失败明确记录，按原文回退实测点。
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    gp.fit(x[:, None], y)
                mean, sd = gp.predict(weeks[:, None], return_std=True)
                lower = mean - 1.96 * sd
                hits = np.flatnonzero(lower >= THRESHOLD)
                preliminary = weeks[hits[0]] if len(hits) else np.inf
                fitted_kernel = str(gp.kernel_)
                warning_text = '; '.join(sorted({str(w.message) for w in caught}))
                method = 'GPR预测区间下界'
                curves.extend({'person_id': pid, 'week': t, 'mean': m, 'sd': s, 'lower95': lo}
                              for t, m, s, lo in zip(weeks, mean, sd, lower))
            except (ValueError, np.linalg.LinAlgError) as exc:
                method = 'GPR失败回退实测'
                warning_text = repr(exc)
        observed = summary.loc[pid, 'first_observed_hit']
        final = max(10., min(preliminary, observed))
        chosen = '实测达标修正' if observed < preliminary else method if np.isfinite(final) else '未达标'
        result.append({'person_id': pid, 'mean_bmi': group.bmi.mean(), 'n_records': len(group),
                       'correlation': corr, 'gpr_preliminary': preliminary,
                       'observed_first': observed, 't_attain': final,
                       'finite': bool(np.isfinite(final)), 'selected_by': chosen,
                       'last_week': group.week.max()})
        logs.append({'person_id': pid, 'branch': method, 'kernel': fitted_kernel, 'warnings': warning_text})
    return pd.DataFrame(result), pd.DataFrame(curves), pd.DataFrame(logs)


def measurement_sensitivity(male, strategies, repeats=100, seed=42):
    """加性读数误差的实测达标诊断，固定原分组；不伪称GPR重拟合结果。"""
    rng = np.random.default_rng(seed)
    obs = [(pid, g.week.to_numpy(), g.y.to_numpy()) for pid, g in male.groupby('person_id', sort=True)]
    bmi = male.groupby('person_id').bmi.mean().reindex([x[0] for x in obs]).to_numpy()
    records = []
    chosen = strategies[(strategies.solver == 'GA') & (strategies.k == 3)]
    for noise in [0., 0.002, 0.005]:
        for repeat in range(1 if noise == 0 else repeats):
            values = []
            for _, t, y in obs:
                perturbed = np.clip(y + rng.normal(0, noise, len(y)), 0., 1.)
                hit = t[perturbed >= THRESHOLD]
                values.append(np.min(hit) if len(hit) else np.inf)
            values = np.asarray(values)
            for row in chosen.itertuples():
                select = (bmi > row.bmi_lower) & (bmi <= row.bmi_upper)
                v = np.sort(values[select])
                # empirical inverse CDF保留无穷；包含原GPR优化排除的孕妇。
                q = v[max(0, int(np.ceil(len(v) * row.target / 100)) - 1)]
                records.append({'sigma_y': noise, 'repeat': repeat, 'target': row.target,
                                'group': row.group, 'n_all': len(v), 'n_censored': int(np.isinf(v).sum()),
                                'observed_quantile': q, 'coverage_at_gpr_week': np.mean(v <= row.week),
                                'method': '固定BMI边界；扰动实测Y；未重拟合GPR'})
    raw = pd.DataFrame(records)
    stats = raw.groupby(['sigma_y', 'target', 'group']).agg(
        coverage_mean=('coverage_at_gpr_week', 'mean'), coverage_sd=('coverage_at_gpr_week', 'std'),
        censored_mean=('n_censored', 'mean'), repetitions=('repeat', 'size')).reset_index()
    finite = raw.assign(finite=raw.observed_quantile.replace(np.inf, np.nan),
                        infeasible=np.isinf(raw.observed_quantile))
    extra = finite.groupby(['sigma_y', 'target', 'group']).agg(
        finite_week_median=('finite', 'median'), infeasible_fraction=('infeasible', 'mean')).reset_index()
    return raw, stats.merge(extra, on=['sigma_y', 'target', 'group'])
