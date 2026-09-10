"""问题2.1/3：重新训练的决策树与论文固定阈值严格分开（p18-25）。"""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, cut_tree
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from .constants import COMPONENTS, SEED, PAPER_THRESHOLDS, PAPER_UNKNOWN
from .data import save_table


# 直接应用论文的PbO固定阈值
def paper_rule(pbo, weather):
    pbo, weather = np.broadcast_arrays(np.asarray(pbo, float), np.asarray(weather))
    if not np.isin(weather, list(PAPER_THRESHOLDS)).all():
        raise ValueError('风化状态必须为风化或无风化。')
    threshold = np.where(weather == '风化', PAPER_THRESHOLDS['风化'], PAPER_THRESHOLDS['无风化'])
    return np.where(pbo <= threshold, '高钾', '铅钡')


# 创建一个使用信息熵准则的决策树
def new_tree(seed=SEED):
    # 连续阈值、熵准则的二叉树；sklearn不是原文声称的纯ID3/SPSSPRO实现。
    return DecisionTreeClassifier(criterion='entropy', random_state=seed)


# 按风化状态分别训练使用全部14种成分的决策树
def fitted_models(data, seed=SEED):
    models = {}
    for weather in PAPER_THRESHOLDS:
        frame = data.samples.loc[data.samples['论文风化'].eq(weather)]
        models[weather] = new_tree(seed).fit(data.matrix(frame), frame['类型'])
    return models


# 按风化状态分别训练仅使用PbO的决策树
def fitted_pbo_models(data, seed=SEED):
    """按论文已选定的PbO特征重训阈值；特征选择来自原文，不是自动选择。"""
    models = {}
    for weather in PAPER_THRESHOLDS:
        frame = data.samples.loc[data.samples['论文风化'].eq(weather)]
        models[weather] = new_tree(seed).fit(frame[['PbO']], frame['类型'])
    return models


# 计算分类准确率及宏平均精确率、召回率和F1
def _metrics(y, pred):
    return {'accuracy': accuracy_score(y, pred),
            'precision_macro': precision_score(y, pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y, pred, average='macro', zero_division=0),
            'F1_macro': f1_score(y, pred, average='macro', zero_division=0)}


# 训练和评估决策树并保存分类规则与数据划分记录
def analyze_trees(data, output, seed=SEED):
    rows, splits, rules = [], [], []
    for weather in PAPER_THRESHOLDS:
        frame = data.samples.loc[data.samples['论文风化'].eq(weather)].reset_index(drop=True)
        x, y, ids = data.matrix(frame), frame['类型'].to_numpy(), frame['文物编号'].to_numpy()
        train, test = train_test_split(np.arange(len(frame)), test_size=.3, random_state=seed, stratify=y)
        model = new_tree(seed).fit(x[train], y[train])
        overlap = sorted(set(ids[train]) & set(ids[test]))
        for name, idx in [('训练集', train), ('测试集', test)]:
            rows.append({'风化口径': '论文风化', '风化': weather, '评估': '70/30采样点划分',
                         '集合': name, 'n': len(idx), 'seed': seed,
                         '训练测试重叠文物': '|'.join(overlap), **_metrics(y[idx], model.predict(x[idx]))})
        rules.append({'风化': weather, '模型': '70/30训练子集_全部14成分',
                      '规则': export_text(model, feature_names=COMPONENTS)})
        pbo_model = new_tree(seed).fit(frame.iloc[train][['PbO']], y[train])
        for name, idx in [('训练集', train), ('测试集', test)]:
            rows.append({'风化口径': '论文风化', '风化': weather, '评估': '70/30采样点划分_预选PbO',
                         '集合': name, 'n': len(idx), 'seed': seed,
                         '训练测试重叠文物': '|'.join(overlap),
                         **_metrics(y[idx], pbo_model.predict(frame.iloc[idx][['PbO']]))})
        rules.append({'风化': weather, '模型': '70/30训练子集_预选PbO',
                      '规则': export_text(pbo_model, feature_names=['PbO'])})
        for fold, (tr, te) in enumerate(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(x, y, ids), 1):
            assert not (set(ids[tr]) & set(ids[te]))
            m = new_tree(seed).fit(x[tr], y[tr])
            rows.append({'风化口径': '论文风化', '风化': weather, '评估': '按文物分组5折CV',
                         '集合': f'第{fold}折测试集', 'n': len(te), 'seed': seed,
                         '训练测试重叠文物': '', **_metrics(y[te], m.predict(x[te]))})
            pbo_model = new_tree(seed).fit(frame.iloc[tr][['PbO']], y[tr])
            rows.append({'风化口径': '论文风化', '风化': weather, '评估': '按文物分组5折CV_预选PbO',
                         '集合': f'第{fold}折测试集', 'n': len(te), 'seed': seed,
                         '训练测试重叠文物': '', **_metrics(y[te], pbo_model.predict(frame.iloc[te][['PbO']]))})
            for name, idx in [('train', tr), ('test', te)]:
                for i in idx:
                    splits.append({'风化': weather, 'fold': fold, 'set': name,
                                   '文物编号': ids[i], '文物采样点': frame.loc[i, '文物采样点']})
    for weather, model in fitted_models(data, seed).items():
        rules.append({'风化': weather, '模型': '全部有效采样点_用于预测',
                      '规则': export_text(model, feature_names=COMPONENTS)})
    for weather, model in fitted_pbo_models(data, seed).items():
        rules.append({'风化': weather, '模型': '全部有效采样点_预选PbO',
                      '规则': export_text(model, feature_names=['PbO'])})
    save_table(pd.DataFrame(rows), output, '21_决策树评估')
    save_table(pd.DataFrame(splits), output, '21_分组交叉验证划分')
    save_table(pd.DataFrame(rules), output, '21_训练树规则')
    frame = data.samples[['文物采样点', '类型', '论文风化']].copy()
    frame['论文固定规则预测'] = paper_rule(data.samples['PbO'], data.samples['论文风化'])
    save_table(frame, output, '21_论文固定阈值_训练样本对照')
    return pd.DataFrame(rows)


# 分类未知样本并计算其PbO含量到论文阈值的距离
def classify_unknown(data, seed=SEED):
    frame = data.unknown.copy()
    if not frame['有效'].all():
        raise ValueError('表单3中存在无效成分合计，停止分类，请先检查附件。')
    models = fitted_models(data, seed)
    pbo_models = fitted_pbo_models(data, seed)
    frame['论文固定规则'] = paper_rule(frame['PbO'], frame['表面风化'])
    frame['PbO重训树预测'] = ''
    frame['重训树预测'] = ''
    frame['Q聚类对照'] = ''
    for weather, model in models.items():
        mask = frame['表面风化'].eq(weather)
        subset = frame.loc[mask]
        frame.loc[mask, '重训树预测'] = model.predict(data.matrix(subset))
        frame.loc[mask, 'PbO重训树预测'] = pbo_models[weather].predict(subset[['PbO']])
        known = data.samples.loc[data.samples['论文风化'].eq(weather)]
        x = np.vstack([data.matrix(known), data.matrix(subset)])
        labels = cut_tree(linkage(x, method='ward'), n_clusters=[2]).ravel()
        # 只以已知样本的多数标签命名簇；未知标签绝不能参与簇命名。
        label_map = {}
        for cluster in np.unique(labels):
            known_labels = known.iloc[np.flatnonzero(labels[:len(known)] == cluster)]['类型']
            label_map[cluster] = known_labels.mode().iloc[0] if len(known_labels) else '无已知样本可命名'
        frame.loc[mask, 'Q聚类对照'] = [label_map[c] for c in labels[len(known):]]
    frame['论文表26标签_参考'] = frame['文物编号'].map(PAPER_UNKNOWN)
    frame['阈值_原始百分比'] = frame['表面风化'].map(PAPER_THRESHOLDS)
    frame['距阈值_百分点'] = frame['PbO'] - frame['阈值_原始百分比']
    # 零含量的乘性扰动不动；另报达到决策边界需要的绝对变化。
    frame['PbO乘性变化到边界'] = np.where(frame['PbO'] > 0,
                                           abs(frame['距阈值_百分点']) / frame['PbO'].replace(0, np.nan), np.nan)
    return frame


# 随机扰动未知样本的成分含量并统计分类标签保持率
def unknown_sensitivity(data, repeats=1000, seed=SEED):
    if repeats < 1:
        raise ValueError('扰动次数必须为正。')
    models = fitted_models(data, seed)
    pbo_models = fitted_pbo_models(data, seed)
    base = classify_unknown(data, seed)
    rng = np.random.default_rng(seed)
    rows = []
    for _, sample in base.iterrows():
        x = sample[COMPONENTS].to_numpy(float)
        weather = sample['表面风化']
        for amplitude in [0, .05, .10, .20, .50, 1.0]:
            # 同时独立扰动全部原始含量；保留原始百分比空间，不重新闭合改写阈值含义。
            perturbed = x * rng.uniform(1 - amplitude, 1 + amplitude, (repeats, len(COMPONENTS)))
            rule = paper_rule(perturbed[:, COMPONENTS.index('PbO')], weather)
            trained = models[weather].predict(perturbed)
            pbo_pred = pbo_models[weather].predict(pd.DataFrame(perturbed[:, [8]], columns=['PbO']))
            rows.append({'文物编号': sample['文物编号'], '扰动比例': amplitude, '重复次数': repeats,
                         'seed': seed, '论文规则标签保持率': np.mean(rule == sample['论文固定规则']),
                         '重训树标签保持率': np.mean(trained == sample['重训树预测']),
                         'PbO重训树标签保持率': np.mean(pbo_pred == sample['PbO重训树预测']),
                         '解释': '有限次乘性扰动；零值不变；保持率不是准确率'})
    return pd.DataFrame(rows)


# 执行未知样本分类和敏感性分析并保存结果
def analyze_unknown(data, output, repeats=1000, seed=SEED):
    result = classify_unknown(data, seed)
    save_table(result, output, '31_未知样本分类与边界距离')
    save_table(unknown_sensitivity(data, repeats, seed), output, '32_未知样本敏感性')
    return result
