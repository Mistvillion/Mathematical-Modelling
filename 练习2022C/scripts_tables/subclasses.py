"""问题2.2：R型变量聚类、Q型样本聚类与有限扰动检验（p20-24）。

图7可见高钾风化组只有6点，因此四大组采用表单1文物风化，
而不是表9/10用于决策树的论文推定风化。原文未交代聚类输入空间，
本实现明确使用原始百分比、不标准化；重新计算结果不冒充表21-24。
"""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, cut_tree
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score, silhouette_score
from .constants import COMPONENTS, REPRESENTATIVES, PAPER_PERTURBATIONS, SEED
from .data import save_table
from .weathering import ward_labels


# 按相关距离对非恒定成分做R型聚类并选取各类代表
def r_cluster(x):
    """相关距离1-|r| + single linkage，常量列不强行赋相关系数。

    在每个R类中选择到同类其余变量总距离最小的代表；并列按附件列序。
    这是补写的确定性选择规则，原文没有交代选代表的可执行规则。
    """
    x = np.asarray(x, float)
    variable = np.std(x, axis=0) > 1e-12
    names = np.asarray(COMPONENTS)[variable]
    corr = np.corrcoef(x[:, variable], rowvar=False)
    if len(names) < 2:
        raise ValueError('非恒定成分过少，无法进行R型聚类。')
    distance = np.clip(1 - abs(corr), 0, 1)
    np.fill_diagonal(distance, 0)
    distance = (distance + distance.T) / 2
    z = linkage(squareform(distance, checks=False), method='single')
    labels = cut_tree(z, n_clusters=[min(3, len(names))]).ravel()
    chosen, rows = [], []
    for label in np.unique(labels):
        loc = np.flatnonzero(labels == label)
        representative = names[loc[np.argmin(distance[np.ix_(loc, loc)].sum(axis=1))]]
        chosen.append(str(representative))
        for i in loc:
            rows.append({'成分': names[i], 'R类': label + 1, '自动代表': names[i] == representative, '状态': '参与'})
    for name in np.asarray(COMPONENTS)[~variable]:
        rows.append({'成分': name, 'R类': np.nan, '自动代表': False, '状态': '恒定列_相关未定义'})
    return z, names.tolist(), pd.DataFrame(rows), chosen


# 按玻璃类型和文物风化状态筛选样本
def subgroup(data, glass, weather):
    return data.samples.loc[data.samples['类型'].eq(glass) & data.samples['文物风化'].eq(weather)].copy()


# 使用论文选定的代表成分将指定样本组划分为两个亚类
def subclass_result(data, glass, weather):
    frame = subgroup(data, glass, weather)
    features = REPRESENTATIVES[(glass, weather)]
    x = frame[features].to_numpy(float)
    labels = ward_labels(x)
    return frame, features, x, labels


# 扰动指定成分并汇总重新聚类与原分组的一致程度
def perturbation_summary(x, labels, j, amplitude, repeats, rng):
    scores = []
    for _ in range(repeats):
        changed = x.copy()
        changed[:, j] *= rng.uniform(1 - amplitude, 1 + amplitude, len(x))
        scores.append(adjusted_rand_score(labels, ward_labels(changed)))
    a = np.asarray(scores)
    return {'ARI均值': a.mean(), 'ARI最小': a.min(), '完全一致比例': np.mean(np.isclose(a, 1))}


# 执行R型变量聚类、Q型亚类划分和扰动分析并保存结果
def analyze(data, output, repeats=200, seed=SEED):
    if repeats < 1:
        raise ValueError('扰动次数须为正。')
    member_rows, summary_rows, r_rows, stability = [], [], [], []
    rng = np.random.default_rng(seed)
    for (glass, weather), features in REPRESENTATIVES.items():
        frame, _, x, labels = subclass_result(data, glass, weather)
        _, _, rtable, automatic = r_cluster(data.matrix(frame))
        rtable.insert(0, '风化', weather)
        rtable.insert(0, '类型', glass)
        r_rows.append(rtable)
        alternative = ward_labels(frame[automatic].to_numpy(float))
        for i, (_, sample) in enumerate(frame.iterrows()):
            member_rows.append({'类型': glass, '文物风化': weather, '文物采样点': sample['文物采样点'],
                                '论文表25特征': '|'.join(features), 'Q亚类': int(labels[i]) + 1,
                                '自动R代表': '|'.join(automatic), '自动R代表_Q亚类': int(alternative[i]) + 1})
        summary_rows.append({'类型': glass, '风化': weather, 'n': len(frame), 'k': 2,
                             'silhouette_论文特征': silhouette_score(x, labels),
                             '自动特征与论文特征ARI': adjusted_rand_score(labels, alternative),
                             '数据空间': 'raw百分比_不标准化'})
        for j, feature in enumerate(features):
            paper_amplitude = PAPER_PERTURBATIONS[(glass, weather)][j]
            for amplitude in sorted(set([.05, .10, .15, .20, paper_amplitude])):
                score = perturbation_summary(x, labels, j, amplitude, repeats, rng)
                stability.append({'类型': glass, '风化': weather, '扰动成分': feature,
                                  '扰动比例': amplitude, '论文表25声称范围': paper_amplitude,
                                  '重复次数': repeats, 'seed': seed, **score})
    result = pd.DataFrame(member_rows)
    save_table(result, output, '22_Q亚类成员')
    save_table(pd.concat(r_rows, ignore_index=True), output, '22_R变量分组与自动代表')
    save_table(pd.DataFrame(summary_rows), output, '22_亚类合理性指标')
    save_table(pd.DataFrame(stability), output, '23_亚类敏感性')
    return result
