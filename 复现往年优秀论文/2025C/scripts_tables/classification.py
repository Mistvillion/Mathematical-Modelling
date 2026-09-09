"""问题四：四个LightGBM专家 + 正文9维元模型 + Z规则。

外层按孕妇分组评估；内层调参、生成OOF元特征。所有指标直接来自附件标签AB。
不合并缺失的作者中间表，不使用孕妇编号、序号或出生结局作为特征。
"""
from dataclasses import dataclass
import json
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             classification_report, log_loss, roc_auc_score,
                             precision_score, recall_score, f1_score, average_precision_score)
from .constants import LABELS

EXPERTS = ['t13', 't18', 't21', 'interaction']
FEATURES = ['age', 'height', 'weight', 'bmi', 'week', 'reads', 'mapping', 'duplicate',
            'unique_reads', 'unique_ratio', 'gc', 'z13', 'z18', 'z21', 'zx', 'x',
            'gc13', 'gc18', 'gc21', 'filtered', 'ivf', 'pregnancies', 'parity']
META_RAW = ['age', 'bmi', 'z13', 'z18', 'z21']


@dataclass
class Fitted:
    estimator: object
    medians: pd.Series
    constant: int | None = None

    def probabilities(self, x, n_classes):
        result = np.zeros((len(x), n_classes))
        if self.constant is not None:
            result[:, self.constant] = 1.
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='X does not have valid feature names')
                probabilities = self.estimator.predict_proba(x.fillna(self.medians).to_numpy())
            result[:, self.estimator.classes_.astype(int)] = probabilities
        return result


def fit_model(x, y, params, seed, validation=None):
    medians = x.median().fillna(0.)
    if len(np.unique(y)) == 1:
        return Fitted(None, medians, int(np.asarray(y)[0]))
    classifier = lgb.LGBMClassifier(**params, n_jobs=1, verbosity=-1,
                                    random_state=seed, deterministic=True, force_col_wise=True)
    kwargs = {}
    if validation is not None:
        vx, vy = validation
        # 训练折缺少稀有类别时，外部标签不能交给LightGBM的编码器。
        if set(np.unique(vy)) <= set(np.unique(y)):
            kwargs = {'eval_set': [(vx.fillna(medians).to_numpy(), np.asarray(vy))],
                      'callbacks': [lgb.early_stopping(25, verbose=False)]}
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message="The argument 'eval_set' is deprecated")
        classifier.fit(x.fillna(medians).to_numpy(), np.asarray(y), **kwargs)
    return Fitted(classifier, medians)


def group_splits(frame, folds, seed):
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    splits = list(splitter.split(frame, frame.abnormal, frame.person_id))
    for train, valid in splits:
        assert set(frame.iloc[train].person_id).isdisjoint(frame.iloc[valid].person_id)
    return splits


def tune(x, y, splits, trials, seed, n_classes, max_estimators, label, outer):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    trial_rows = []
    def objective(trial):
        params = {'learning_rate': trial.suggest_float('learning_rate', .01, .2 if n_classes > 2 else .1),
                  'num_leaves': trial.suggest_int('num_leaves', 20 if n_classes > 2 else 10, 200 if n_classes > 2 else 150),
                  'n_estimators': max_estimators}
        if n_classes > 2:
            params['max_depth'] = trial.suggest_int('max_depth', 3, 10)
        else:
            params['is_unbalance'] = True
        scores, iterations = [], []
        for index, (train, valid) in enumerate(splits):
            fitted = fit_model(x.iloc[train], y.iloc[train], params, seed + index,
                               validation=(x.iloc[valid], y.iloc[valid]))
            predicted = fitted.probabilities(x.iloc[valid], n_classes)
            scores.append(log_loss(y.iloc[valid], np.clip(predicted, 1e-10, 1 - 1e-10) /
                                   np.clip(predicted, 1e-10, 1 - 1e-10).sum(axis=1, keepdims=True), labels=np.arange(n_classes)))
            iterations.append(fitted.estimator.best_iteration_ or max_estimators if fitted.estimator is not None else 1)
        trial.set_user_attr('n_estimators', max(1, int(np.median(iterations))))
        value = float(np.mean(scores))
        trial_rows.append({'outer_fold': outer, 'model': label, 'trial': trial.number,
                           'log_loss': value, 'params': json.dumps(params, sort_keys=True),
                           'refit_estimators': trial.user_attrs['n_estimators']})
        return value
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=trials)
    best = dict(study.best_params, n_estimators=study.best_trial.user_attrs['n_estimators'])
    if n_classes == 2:
        best['is_unbalance'] = True
    return best, trial_rows


def meta_input(frame, expert_probabilities):
    result = pd.DataFrame(expert_probabilities, columns=[f'p_{name}' for name in EXPERTS], index=frame.index)
    return pd.concat([result, frame[META_RAW]], axis=1)


def train_stack(frame, args, outer):
    splits = group_splits(frame, args.inner_folds, args.seed + outer)
    x = frame[FEATURES]
    oof = np.zeros((len(frame), 4))
    models, history, params, importances = [], [], [], []
    for j, target in enumerate(EXPERTS):
        best, records = tune(x, frame[target], splits, args.trials, args.seed + outer * 17 + j,
                             2, args.max_estimators, target, outer)
        history.extend(records)
        params.append({'outer_fold': outer, 'model': target, 'params': json.dumps(best, sort_keys=True)})
        for train, valid in splits:
            expert = fit_model(x.iloc[train], frame[target].iloc[train], best, args.seed)
            oof[valid, j] = expert.probabilities(x.iloc[valid], 2)[:, 1]
        expert = fit_model(x, frame[target], best, args.seed)
        models.append(expert)
        if expert.estimator is not None:
            gain = expert.estimator.booster_.feature_importance(importance_type='gain')
            for feature, value in zip(FEATURES, gain):
                importances.append({'outer_fold': outer, 'model': target, 'feature': feature, 'gain': value})
    xm = meta_input(frame, oof)
    best, records = tune(xm, frame.state, splits, args.meta_trials, args.seed + outer * 31,
                         8, args.max_estimators, 'meta', outer)
    history.extend(records)
    params.append({'outer_fold': outer, 'model': 'meta', 'params': json.dumps(best, sort_keys=True)})
    meta = fit_model(xm, frame.state, best, args.seed)
    return models, meta, history, params, importances, oof


def predict_stack(frame, models, meta):
    expert = np.column_stack([model.probabilities(frame[FEATURES], 2)[:, 1] for model in models])
    probabilities = meta.probabilities(meta_input(frame, expert), 8)
    primary = np.argmax(probabilities, axis=1)
    logic = primary.copy()
    first = (frame.z13 >= 3) & (frame.z21 >= 3)
    second = ~first & (frame.z18 >= 3) & (frame.z21 >= 3)
    logic[first] = 5
    logic[second] = 6
    return primary, logic, probabilities, expert


def binary_metrics(truth, predictions, probabilities, scope, variant):
    abnormal = np.asarray(truth) > 0
    predicted = np.asarray(predictions) > 0
    return {'scope': scope, 'variant': variant, 'n': len(truth), 'positives': abnormal.sum(),
            'accuracy': accuracy_score(abnormal, predicted), 'balanced_accuracy': balanced_accuracy_score(abnormal, predicted),
            'precision': precision_score(abnormal, predicted, zero_division=0),
            'recall': recall_score(abnormal, predicted, zero_division=0),
            'f1': f1_score(abnormal, predicted, zero_division=0),
            'roc_auc': roc_auc_score(abnormal, probabilities) if len(np.unique(abnormal)) > 1 else np.nan,
            'average_precision': average_precision_score(abnormal, probabilities) if abnormal.any() else np.nan,
            'multiclass_accuracy': accuracy_score(truth, predictions)}


def analyse(data, args):
    frame = data.female.reset_index(drop=True)
    rows, histories, params, importances, fold_records, meta_records, expert_metrics = [], [], [], [], [], [], []
    for fold, (train, test) in enumerate(group_splits(frame, args.folds, args.seed), 1):
        print(f'  问题4：孕妇分组外层折 {fold}/{args.folds}，训练{len(train)}条，验证{len(test)}条', flush=True)
        training, testing = frame.iloc[train], frame.iloc[test]
        models, meta, trials, settings, gains, oof = train_stack(training, args, fold)
        histories.extend(trials)
        params.extend(settings)
        importances.extend(gains)
        primary, logic, proba, expert = predict_stack(testing, models, meta)
        for i, record in enumerate(testing.itertuples()):
            row = {'row_id': record.row_id, 'person_id': record.person_id, 'fold': fold,
                   'true_state': record.state, 'predicted_state': int(primary[i]),
                   'logic_state': int(logic[i]), 'p_abnormal': 1 - proba[i, 0]}
            row.update({f'p_state{j}': proba[i, j] for j in range(8)})
            row.update({f'p_{target}': expert[i, j] for j, target in enumerate(EXPERTS)})
            rows.append(row)
        for role, part in [('train', training), ('test', testing)]:
            fold_records.extend({'outer_fold': fold, 'role': role, 'person_id': pid} for pid in sorted(part.person_id.unique()))
        for i, record in enumerate(training.itertuples()):
            meta_records.append({'outer_fold': fold, 'row_id': record.row_id, 'person_id': record.person_id,
                                 **{f'p_{target}': oof[i, j] for j, target in enumerate(EXPERTS)}})
        for j, target in enumerate(EXPERTS):
            truth = testing[target].to_numpy()
            expert_metrics.append({'outer_fold': fold, 'model': target, 'n': len(testing),
                                   'positives': int(truth.sum()), 'log_loss': log_loss(truth, expert[:, j], labels=[0, 1]),
                                   'auc': roc_auc_score(truth, expert[:, j]) if len(np.unique(truth)) > 1 else np.nan,
                                   'recall': recall_score(truth, expert[:, j] >= .5, zero_division=0)})
    predicted = pd.DataFrame(rows).sort_values('row_id').reset_index(drop=True)
    evaluations, reports, matrices = [], [], []
    for scope, part in [('OOF_all', predicted)] + [(f'fold_{k}', v) for k, v in predicted.groupby('fold')]:
        for variant, col in [('stack', 'predicted_state'), ('stack_plus_z_rule', 'logic_state')]:
            evaluations.append(binary_metrics(part.true_state, part[col], part.p_abnormal, scope, variant))
    for variant, col in [('stack', 'predicted_state'), ('stack_plus_z_rule', 'logic_state')]:
        report = classification_report(predicted.true_state, predicted[col], labels=list(range(7)),
                                       target_names=[LABELS[i] for i in range(7)], output_dict=True, zero_division=0)
        for label, value in report.items():
            if isinstance(value, dict):
                reports.append({'variant': variant, 'label': label, **value})
        for kind, truth, prediction, labels in [
                ('multiclass', predicted.true_state, predicted[col], list(range(8))),
                ('binary', (predicted.true_state > 0).astype(int), (predicted[col] > 0).astype(int), [0, 1])]:
            cm = confusion_matrix(truth, prediction, labels=labels)
            for i, real in enumerate(labels):
                for j, guess in enumerate(labels):
                    matrices.append({'variant': variant, 'kind': kind, 'true': real, 'predicted': guess, 'count': cm[i, j]})
    # 无异常基线与规则基线，避免单看不平衡数据的accuracy。
    evaluations.append(binary_metrics(predicted.true_state, np.zeros(len(predicted), int),
                                       np.zeros(len(predicted)), 'OOF_all', 'all_normal_baseline'))
    z = frame.set_index('row_id').loc[predicted.row_id, ['z13', 'z18', 'z21']].to_numpy()
    rule = ((z[:, 0] >= 3) + 2 * (z[:, 1] >= 3) + 4 * (z[:, 2] >= 3)).astype(int)
    evaluations.append(binary_metrics(predicted.true_state, rule, (rule > 0).astype(float), 'OOF_all', 'z3_baseline'))
    # 按孕妇重采样置信区间，保留一位孕妇全部记录。
    rng = np.random.default_rng(args.seed)
    groups = [g.index.to_numpy() for _, g in predicted.groupby('person_id')]
    bootstrap = []
    for repeat in range(args.sensitivity_repeats):
        selected = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        part = predicted.loc[selected]
        stats = binary_metrics(part.true_state, part.predicted_state, part.p_abnormal, 'OOF_bootstrap', 'stack')
        bootstrap.append({'repeat': repeat, **stats})
    boot = pd.DataFrame(bootstrap)
    intervals = []
    for metric in ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'multiclass_accuracy']:
        intervals.append({'metric': metric, 'lower_2.5': boot[metric].quantile(.025),
                          'upper_97.5': boot[metric].quantile(.975), 'repeats': len(boot)})
    return {'41_孕妇分组验证划分': pd.DataFrame(fold_records), '41_折内OOF元特征': pd.DataFrame(meta_records),
            '41_Optuna搜索记录': pd.DataFrame(histories), '41_最佳超参数': pd.DataFrame(params),
            '41_专家特征重要性': pd.DataFrame(importances), '41_专家验证指标': pd.DataFrame(expert_metrics),
            '42_女胎逐记录OOF预测': predicted, '42_总体与分折指标': pd.DataFrame(evaluations),
            '42_分类报告': pd.DataFrame(reports), '42_混淆矩阵': pd.DataFrame(matrices),
            '42_孕妇Bootstrap区间': pd.DataFrame(intervals)}
