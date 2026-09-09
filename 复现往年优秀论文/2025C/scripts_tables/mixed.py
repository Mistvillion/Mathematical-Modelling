"""问题一：Pearson、随机截距LMM、ML比较与REML诊断。"""
from itertools import product
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan
from .constants import PAPER_COEF


def design(male):
    d = male.copy()
    for col in ['height', 'weight']:
        d[f'{col}_z'] = (d[col] - d[col].mean()) / d[col].std(ddof=0)
    d['week_sq'] = d.week ** 2
    d['bmi_sq'] = d.bmi ** 2
    d['week_bmi'] = d.week * d.bmi
    return d


def fit_mixed(formula, d, reml=True):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = smf.mixedlm(formula, d, groups=d.person_id).fit(reml=reml, method=['lbfgs', 'bfgs'], disp=False)
    if not result.converged:
        raise RuntimeError(f'LMM未收敛: {formula}')
    # 比例响应导致绝对随机方差<0.01，statsmodels可能发边界提示；记录而不全局屏蔽。
    return result, '; '.join(sorted({str(w.message) for w in caught}))


def analyse(data):
    d = design(data.male)
    cols = ['y', 'week', 'bmi', 'age', 'height_z', 'weight_z', 'draw', 'week_sq', 'bmi_sq', 'week_bmi']
    correlations = []
    for col in cols[1:]:
        r, p = stats.pearsonr(d.y, d[col])
        correlations.append({'feature': col, 'r': r, 'p_naive': p, 'n_records': len(d),
                             'note': '记录层描述相关；重复测量的推断以LMM为准'})
    formula = 'y ~ week + week_sq + height_z + bmi'
    reml, note = fit_mixed(formula, d)
    ml, _ = fit_mixed(formula, d, reml=False)
    ols = smf.ols(formula, d).fit()
    # Stata的固定效应推断使用GLS协方差(X'V^-1X)^-1。
    # statsmodels默认Hessian会耦合方差参数的不确定性，两种口径都保留。
    info = np.zeros((len(reml.fe_params), len(reml.fe_params)))
    sigma2, tau2 = reml.scale, reml.cov_re.iloc[0, 0]
    for x in reml.model.exog_li:
        info += x.T @ x / sigma2 - tau2 / (sigma2 * (sigma2 + len(x) * tau2)) * np.outer(x.sum(axis=0), x.sum(axis=0))
    gls_cov = np.linalg.inv(info)
    coef = []
    for j, term in enumerate(reml.fe_params.index):
        se = np.sqrt(gls_cov[j, j])
        z = reml.fe_params[term] / se
        coef.append({'term': term, 'estimate': reml.fe_params[term], 'se': se,
                     'z': z, 'p': 2 * stats.norm.sf(abs(z)),
                     'ci_low': reml.fe_params[term] - 1.959963984540054 * se,
                     'ci_high': reml.fe_params[term] + 1.959963984540054 * se,
                     'se_observed_hessian': reml.bse_fe[term], 'p_observed_hessian': reml.pvalues[term],
                     'paper_estimate': PAPER_COEF[term], 'difference': reml.fe_params[term] - PAPER_COEF[term]})
    random_var = float(reml.cov_re.iloc[0, 0])
    residual_var = float(reml.scale)
    lr = max(0., 2 * (ml.llf - ols.llf))
    fixed = reml.fe_params.drop('Intercept')
    wald = float(fixed @ np.linalg.solve(gls_cov[1:, 1:], fixed))
    reffects = pd.DataFrame([{'person_id': pid, 'random_intercept': value.iloc[0]}
                            for pid, value in reml.random_effects.items()])
    fitted = reml.fittedvalues
    residuals = pd.DataFrame({'person_id': d.person_id, 'row_id': d.row_id, 'y': d.y,
                             'fitted': fitted, 'fixed_fitted': reml.model.exog @ reml.fe_params,
                             'residual': d.y - fitted, 'standardized_residual': (d.y - fitted) / np.sqrt(residual_var)})
    bp = het_breuschpagan(residuals.residual, reml.model.exog)
    shapiro = stats.shapiro(reffects.random_intercept)
    diagnostics = pd.DataFrame([
        {'metric': 'ICC_REML', 'value': random_var / (random_var + residual_var), 'p': np.nan},
        {'metric': 'random_variance_REML', 'value': random_var, 'p': np.nan},
        {'metric': 'residual_variance_REML', 'value': residual_var, 'p': np.nan},
        {'metric': 'LR_ML_mixture_chi2', 'value': lr, 'p': 0.5 * stats.chi2.sf(lr, 1)},
        {'metric': 'Wald_fixed_4df', 'value': wald, 'p': stats.chi2.sf(wald, 4)},
        {'metric': 'Shapiro_BLUP_exploratory', 'value': shapiro.statistic, 'p': shapiro.pvalue},
        {'metric': 'Breusch_Pagan_exploratory', 'value': bp[0], 'p': bp[1]},
        {'metric': 'turning_week', 'value': -reml.fe_params['week'] / (2 * reml.fe_params['week_sq']), 'p': np.nan},
    ])
    comparisons = pd.DataFrame([
        {'model': '随机截距LMM_ML', 'n': len(d), 'parameters': len(ml.params) + 1, 'll': ml.llf, 'aic': ml.aic, 'bic': ml.bic},
        # OLS的scale也计为一个参数，与Stata mixed无随机效应版本口径一致。
        {'model': '固定效应_ML', 'n': len(d), 'parameters': len(ols.params) + 1, 'll': ols.llf,
         'aic': -2 * ols.llf + 2 * (len(ols.params) + 1),
         'bic': -2 * ols.llf + np.log(len(d)) * (len(ols.params) + 1)},
    ])
    candidates = []
    for use_sq, use_bsq, use_interact in product([False, True], repeat=3):
        terms = ['week', 'bmi'] + (['week_sq'] if use_sq else []) + (['bmi_sq'] if use_bsq else []) + (['week_bmi'] if use_interact else [])
        fit, _ = fit_mixed('y ~ ' + ' + '.join(terms), d, reml=False)
        candidates.append({'formula': ' + '.join(terms), 'aic_ml': fit.aic, 'bic_ml': fit.bic,
                           'll_ml': fit.llf, 'converged': fit.converged})
    return {'11_Pearson相关': pd.DataFrame(correlations), '11_相关矩阵': d[cols].corr().rename_axis('feature').reset_index(),
            '12_固定效应与论文表1对照': pd.DataFrame(coef), '12_ML模型比较': comparisons,
            '12_八种候选模型': pd.DataFrame(candidates), '13_模型检验': diagnostics,
            '13_随机截距': reffects, '13_残差': residuals,
            '12_拟合日志': pd.DataFrame([{'reml_converged': reml.converged, 'warnings': note,
                                        'height_mean': d.height.mean(), 'height_sd': d.height.std(ddof=0)}])}
