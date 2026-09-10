"""问题1第三小问：Ward分组、四中心二次回归和逆向预测（p11-18）。

默认回归采用论文表13/14的中心顺序；高钾的第3/4阶段与表11相反。
同时输出按表11/12顺序拟合的对照。这里的“阶段”只是作者假设，不能当真实时间。
可运行版修复原附录的阶段3误用4与负数开根号；结果不宣称逐值复现表17/18。
"""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, cut_tree
from sklearn.metrics import adjusted_rand_score
from .constants import COMPONENTS, TYPES
from .composition import paper_clr, inverse_clr
from .data import save_table


# 用Ward方法将样本分成k类
def ward_labels(x, k=2):
    x = np.asarray(x, dtype=float)
    if len(x) < k or k < 1:
        raise ValueError('聚类数必须介于1与样本数之间。')
    if k == 1:
        return np.zeros(len(x), dtype=int)
    return cut_tree(linkage(x, method='ward'), n_clusters=[k]).ravel()


# 先用Ward方法二分样本再将各分支分别二分
def two_then_two(x):
    """对应附录B：先二分，各分支内再二分；不是直接指定四类。"""
    outer = ward_labels(x)
    inner = np.zeros(len(x), dtype=int)
    for cluster in [0, 1]:
        loc = np.flatnonzero(outer == cluster)
        labels = ward_labels(x[loc]) if len(loc) >= 2 else np.zeros(len(loc), dtype=int)
        inner[loc] = cluster * 2 + labels + 1
    return outer, inner


# 根据论文表格顺序确定回归使用的阶段编号
def fit_stages(frame, order='table13_14'):
    """表13的高钾第3/4中心与表11名单相反，显式记录这一原文矛盾。"""
    stages = frame['论文阶段'].to_numpy().copy()
    if order == 'table13_14':
        swapped = frame['类型'].eq('高钾').to_numpy() & (stages >= 3)
        stages[swapped] = 7 - stages[swapped]
    elif order != 'table11_12':
        raise ValueError(f'未知阶段顺序: {order}')
    return stages


# 用四阶段成分中心拟合各类玻璃的二次变化曲线
def fit_trends(data, order='table13_14'):
    rows, coefficients = [], {}
    for glass in TYPES:
        frame = data.samples.loc[data.samples['类型'].eq(glass)]
        z, _ = paper_clr(data.matrix(frame))
        stages = fit_stages(frame, order)
        centers = np.array([z[stages == t].mean(axis=0) for t in range(1, 5)])
        # polyfit以四个阶段等权拟合，对应原文，不以组样本数加权。
        coeff = np.polyfit(np.arange(1, 5), centers, 2).T
        fitted = np.array([np.polyval(c, np.arange(1, 5)) for c in coeff]).T
        ss_res = ((centers - fitted) ** 2).sum(axis=0)
        ss_tot = ((centers - centers.mean(axis=0)) ** 2).sum(axis=0)
        r2 = np.full(len(COMPONENTS), np.nan)
        np.divide(ss_res, ss_tot, out=r2, where=ss_tot > 1e-20)
        r2 = 1 - r2
        coefficients[glass] = coeff
        for j, chemical in enumerate(COMPONENTS):
            rows.append({'类型': glass, '成分': chemical, '阶段依据': order, 'a': coeff[j, 0], 'b': coeff[j, 1],
                         'c': coeff[j, 2], 'R2': r2[j],
                         **{f'阶段{t}中心': centers[t - 1, j] for t in range(1, 5)}})
    return pd.DataFrame(rows), coefficients


# 将样本对数比坐标推回第一阶段并还原风化前含量
def backcast(z, stages, coeff, present, skipped=(), method='signed_sqrt'):
    """y0=f(1)+g(y-f(t))。shift是平移模型；signed_sqrt是有符号开根残差。

    原代码以y的符号分支、且阶段3也代入4，可能生成复数。这里按残差符号
    修正，所有零位由原始检测mask保留。被跳过的是CLR坐标，闭合后含量仍可能变。
    """
    z = np.asarray(z, float)
    stages = np.asarray(stages, int)
    baseline = coeff.sum(axis=1)
    fitted = np.array([np.polyval(c, stages) for c in coeff]).T
    residual = z - fitted
    if method == 'signed_sqrt':
        residual = np.sign(residual) * np.sqrt(np.abs(residual))
    elif method != 'shift':
        raise ValueError(f'未知预测方法: {method}')
    predicted_z = baseline + residual
    for chemical in skipped:
        j = COMPONENTS.index(chemical)
        predicted_z[:, j] = z[:, j]
    return inverse_clr(predicted_z, present), predicted_z


# 批量预测风化样本的风化前成分并整理结果表
def predictions(data, method='signed_sqrt'):
    _, coeff = fit_trends(data)
    frames = []
    for glass in TYPES:
        selected = data.samples['类型'].eq(glass) & data.samples['论文阶段'].ge(3)
        frame = data.samples.loc[selected]
        z, present = paper_clr(data.matrix(frame))
        skipped = ['Na2O', 'SnO2', 'SO2'] if glass == '高钾' else ['K2O', 'MgO']
        x0, _ = backcast(z, fit_stages(frame), coeff[glass], present, skipped, method)
        out = frame[['文物采样点', '文物编号', '类型', '论文阶段']].reset_index(drop=True).copy()
        out['拟合阶段_表13_14'] = fit_stages(frame)
        out['预测方法'] = method
        out[COMPONENTS] = x0
        out['含量合计'] = x0.sum(axis=1)
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


# 将原始与预测样本混合聚类以检查成分相似性
def mixed_cluster_check(data, glass):
    """p18/附录A：混入预测数据后平均联接；只是相似性对照，并非外部验证。"""
    frame = data.samples.loc[data.samples['类型'].eq(glass)]
    pred = predictions(data)
    pred = pred.loc[pred['类型'].eq(glass)]
    x = np.vstack([data.matrix(frame, 'closed'), pred[COMPONENTS].to_numpy(float)])
    names = frame['文物采样点'].tolist() + ['*' + s for s in pred['文物采样点']]
    z = linkage(x, method='average')
    labels = cut_tree(z, n_clusters=[2]).ravel()
    return z, pd.DataFrame({'采样点': names, '来源': ['原始'] * len(frame) + ['预测'] * len(pred), '类': labels + 1})


# 执行Ward分组对照、阶段回归和风化前预测并保存结果
def analyze(data, output):
    fits, _ = fit_trends(data)
    save_table(fits, output, '13_回归系数与四阶段中心')
    alternative_fits, _ = fit_trends(data, 'table11_12')
    save_table(alternative_fits, output, '13_按表11_12名单顺序回归_对照')
    checks, summaries = [], []
    for glass in TYPES:
        frame = data.samples.loc[data.samples['类型'].eq(glass)]
        outer, inner = two_then_two(data.matrix(frame, 'raw'))
        check = frame[['文物采样点', '类型', '点位风化', '论文风化', '论文阶段']].copy()
        check['重算Ward二类'] = outer + 1
        check['重算Ward二再二类_无时间含义'] = inner
        checks.append(check)
        summaries.append({'类型': glass, '二类ARI_对论文风化': adjusted_rand_score(frame['论文风化'], outer),
                          '四类ARI_对论文阶段': adjusted_rand_score(frame['论文阶段'], inner)})
        _, mixed = mixed_cluster_check(data, glass)
        save_table(mixed, output, f'13_{glass}_混合聚类检查')
    save_table(pd.concat(checks), output, '13_论文阶段与Ward重算对照')
    save_table(pd.DataFrame(summaries), output, '13_分组ARI')
    result = predictions(data)
    save_table(result, output, '13_风化前预测_signed_sqrt')
    save_table(predictions(data, 'shift'), output, '13_风化前预测_shift')
    return result
