"""问题三：带有效长度的MLP+LSTM+DeepHit，以及静态区间删失对照。

paper_history使用完整轨迹和GPR伪时间，属于回顾性重构。
static_interval仅使用首次登记的四项生理指标和实测删失区间，不使用Y轨迹预测自身标签。
"""
from copy import deepcopy
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from .data import subjects
from .optimization import daily_quantile, optimise

WEEKS = np.arange(10., 41.)
STATIC = ['age', 'height', 'weight', 'bmi']


class HybridSurvivalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.static = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Dropout(.4),
                                    nn.Linear(32, 16), nn.ReLU(), nn.Dropout(.4))
        self.longitudinal = nn.LSTM(2, 16, batch_first=True)
        # 正文两层分析头；最后一格是40周以后，softmax输出的是PMF而非hazard。
        self.head = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Dropout(.4), nn.Linear(32, len(WEEKS) + 1))

    def forward(self, static, sequences, lengths):
        encoded = self.static(static)
        packed = pack_padded_sequence(sequences, lengths.clamp(min=1).cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.longitudinal(packed)
        hidden = hidden[-1] * (lengths > 0).to(static.dtype)[:, None]
        return self.head(torch.cat([encoded, hidden], dim=1))


def loss_function(logits, lower, upper, event, interval=False):
    probabilities = logits.softmax(dim=1)
    cumulative = probabilities.cumsum(dim=1)
    row = torch.arange(len(logits), device=logits.device)
    cdf_upper = cumulative[row, upper]
    cdf_lower = torch.where(lower >= 0, cumulative[row, lower.clamp(min=0)], torch.zeros_like(cdf_upper))
    event_probability = (cdf_upper - cdf_lower).clamp(min=1e-8) if interval else probabilities[row, upper].clamp(min=1e-8)
    tail_probability = (1 - cdf_upper).clamp(min=1e-8)
    nll = -torch.where(event, event_probability.log(), tail_probability.log()).mean()
    if interval:
        # 实测时间是左/区间删失；不强加无法确定的个体先后顺序。
        return nll
    comparable = event[:, None] & (upper[:, None] < upper[None, :])
    differences = cumulative[row, upper][:, None] - cumulative[:, upper].T
    ranking = torch.exp(-differences / .1)[comparable].mean() if comparable.any() else logits.sum() * 0
    return .2 * nll + .8 * ranking


def concordance(times, events, scores, lower=None):
    """scores为预测时间，越小应越早达标；区间对照仅比较不重叠区间。"""
    comparable = events[:, None] & (times[:, None] < (times if lower is None else lower)[None, :])
    if not comparable.any():
        return np.nan
    differences = scores[None, :] - scores[:, None]
    return float((np.sum((differences > 0) & comparable) + .5 * np.sum((differences == 0) & comparable)) / comparable.sum())


def prepare(male, attainment, seed):
    cohort = subjects(male)
    cohort = cohort.merge(attainment[['person_id', 't_attain']], on='person_id', validate='one_to_one')
    event = np.isfinite(cohort.t_attain.to_numpy())
    all_indices = np.arange(len(cohort))
    develop, test = train_test_split(all_indices, test_size=.2, random_state=seed, stratify=event)
    train, valid = train_test_split(develop, test_size=.25, random_state=seed, stratify=event[develop])
    cohort['split'] = 'train'
    cohort.loc[valid, 'split'] = 'validation'
    cohort.loc[test, 'split'] = 'test'
    scaler = StandardScaler().fit(cohort.loc[train, STATIC])
    static = torch.tensor(scaler.transform(cohort[STATIC]), dtype=torch.float32)
    sequences = [male.loc[male.person_id == pid].sort_values(['week', 'row_id'])[['week', 'y']].to_numpy()
                 for pid in cohort.person_id]
    long_scaler = StandardScaler().fit(np.concatenate([sequences[i] for i in train]))
    lengths = np.array([len(s) for s in sequences])
    padded = np.zeros((len(cohort), max(lengths), 2), dtype=np.float32)
    for i, sequence in enumerate(sequences):
        padded[i, :len(sequence)] = long_scaler.transform(sequence)
    tensors = (static, torch.tensor(padded), torch.tensor(lengths, dtype=torch.long))
    scaling = pd.DataFrame({'feature': STATIC + ['sequence_week', 'sequence_y'],
                            'mean_train': np.r_[scaler.mean_, long_scaler.mean_],
                            'scale_train': np.r_[scaler.scale_, long_scaler.scale_]})
    return cohort, tensors, (train, valid, test), scaling, long_scaler


def targets_for(cohort, mode):
    if mode == 'paper_history':
        event = np.isfinite(cohort.t_attain.to_numpy())
        time = np.where(event, cohort.t_attain, cohort.last_week)
        upper = np.clip(np.floor(time).astype(int) - 10, 0, len(WEEKS) - 1)
        lower = upper - 1
    else:
        event = cohort.observed_event.to_numpy(bool)
        time = np.where(event, cohort.interval_upper, cohort.last_week)
        # 每一整数周PMF代表该周区间；上端向上取整，左端向下取整。
        upper = np.clip(np.ceil(time).astype(int) - 10, 0, len(WEEKS) - 1)
        lower = np.clip(np.floor(cohort.interval_lower).astype(int) - 10, -1, len(WEEKS) - 1)
        upper[~event] = np.clip(np.floor(cohort.last_week.to_numpy()[~event]).astype(int) - 10, 0, len(WEEKS) - 1)
    return tuple(torch.tensor(x, dtype=torch.bool if j == 2 else torch.long)
                 for j, x in enumerate((lower, upper, event)))


def select(values, indices):
    return tuple(value[indices] for value in values)


def fit_one(inputs, targets, splits, mode, args):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = HybridSurvivalNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=.5)
    train, validation, _ = splits
    best_loss, best_state, bad_epochs, history = np.inf, None, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        permuted = rng.permutation(train)
        running = 0.
        for start in range(0, len(train), 32):
            batch = permuted[start:start + 32]
            optimizer.zero_grad()
            loss = loss_function(model(*select(inputs, batch)), *select(targets, batch), interval=mode == 'static_interval')
            if not torch.isfinite(loss):
                raise RuntimeError(f'{mode}训练损失非有限值')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            running += loss.item() * len(batch)
        model.eval()
        with torch.no_grad():
            val_loss = loss_function(model(*select(inputs, validation)), *select(targets, validation), interval=mode == 'static_interval').item()
        history.append({'mode': mode, 'epoch': epoch, 'train_loss': running / len(train),
                        'validation_loss': val_loss, 'learning_rate': optimizer.param_groups[0]['lr']})
        scheduler.step(val_loss)
        if val_loss < best_loss - 1e-6:
            best_loss, best_state, bad_epochs = val_loss, deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
        if bad_epochs >= 20:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, pd.DataFrame(history)


def permutation_importance(model, inputs, targets, indices, mode, repeats, seed):
    generator = torch.Generator().manual_seed(seed)
    x = select(inputs, indices)
    y = select(targets, indices)
    rows = []
    with torch.no_grad():
        baseline = loss_function(model(*x), *y, interval=mode == 'static_interval').item()
        for feature in STATIC + ['longitudinal']:
            changes = []
            for _ in range(repeats):
                perm = torch.randperm(len(indices), generator=generator)
                s, seq, lengths = (v.clone() for v in x)
                if feature == 'longitudinal':
                    seq, lengths = seq[perm], lengths[perm]
                else:
                    col = STATIC.index(feature)
                    s[:, col] = s[perm, col]
                loss = loss_function(model(s, seq, lengths), *y, interval=mode == 'static_interval').item()
                changes.append(loss - baseline)
            rows.append({'mode': mode, 'feature': feature, 'loss_increase_mean': np.mean(changes),
                         'loss_increase_sd': np.std(changes), 'n_repeats': repeats, 'split': 'test'})
    return rows


def analyse(data, attainment, args):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cohort, all_inputs, splits, scaling, long_scaler = prepare(data.male, attainment, args.seed)
    histories, predictions, metrics, importances, strategies, audits, traces, noise_rows = [], [], [], [], [], [], [], []
    for mode in ['paper_history', 'static_interval']:
        inputs = all_inputs if mode == 'paper_history' else (all_inputs[0], torch.zeros_like(all_inputs[1]), torch.zeros_like(all_inputs[2]))
        target = targets_for(cohort, mode)
        model, history = fit_one(inputs, target, splits, mode, args)
        histories.append(history)
        with torch.no_grad():
            pmf = model(*inputs).softmax(dim=1).numpy()
        cdf = pmf[:, :-1].cumsum(axis=1)
        expected = pmf @ np.r_[WEEKS, 41.]
        for i, row in cohort.iterrows():
            for j, week in enumerate(WEEKS):
                predictions.append({'mode': mode, 'person_id': row.person_id, 'split': row.split,
                                    'week': week, 'pmf': pmf[i, j], 'cdf': cdf[i, j], 'tail_after40': pmf[i, -1]})
        times = target[1].numpy() + 10
        events = target[2].numpy()
        for name, index in zip(['train', 'validation', 'test'], splits):
            with torch.no_grad():
                loss = loss_function(model(*select(inputs, index)), *select(target, index), interval=mode == 'static_interval').item()
            metrics.append({'mode': mode, 'split': name, 'n': len(index), 'events': events[index].sum(),
                            'loss': loss, 'c_index': concordance(times[index], events[index], expected[index],
                                lower=cohort.interval_lower.to_numpy()[index] if mode == 'static_interval' else None),
                            'interpretation': '完整轨迹重构GPR伪标签，非前瞻精度' if mode == 'paper_history' else '基线静态特征；实测区间删失'})
        importances.extend(permutation_importance(model, inputs, target, splits[2], mode, 10, args.seed))
        strat, audit, trace = optimise(cohort.bmi.to_numpy(), cdf, args.targets, [3], args,
                                      kind='survival', weeks=WEEKS, mode=mode)
        strategies.append(strat)
        audits.append(audit)
        traces.append(trace)
        if mode == 'paper_history':
            rng = np.random.default_rng(args.seed)
            main = strat[strat.solver == 'GA']
            for sigma in [0., .002, .005]:
                for repeat in range(1 if sigma == 0 else args.sensitivity_repeats):
                    perturbed = inputs[1].clone()
                    noise = rng.normal(0, sigma / long_scaler.scale_[1], perturbed[:, :, 1].shape)
                    valid = np.arange(perturbed.shape[1])[None, :] < inputs[2].numpy()[:, None]
                    perturbed[:, :, 1] += torch.tensor(noise * valid, dtype=torch.float32)
                    with torch.no_grad():
                        perturbed_cdf = model(inputs[0], perturbed, inputs[2]).softmax(1)[:, :-1].cumsum(1).numpy()
                    for row in main.itertuples():
                        mask = (cohort.bmi > row.bmi_lower) & (cohort.bmi <= row.bmi_upper)
                        mean = perturbed_cdf[mask].mean(axis=0)
                        week = daily_quantile(mean, WEEKS, row.target / 100)
                        noise_rows.append({'sigma_y': sigma, 'repeat': repeat, 'target': row.target,
                                           'group': row.group, 'week': week, 'base_week': row.week,
                                           'feasible': bool(np.isfinite(week)),
                                           'coverage_at_base': np.interp(row.week, WEEKS, mean) if np.isfinite(row.week) else np.nan})
    noise = pd.DataFrame(noise_rows)
    finite_noise = noise.assign(finite_week=noise.week.replace(np.inf, np.nan),
                               delta=(noise.week - noise.base_week).replace([np.inf, -np.inf], np.nan))
    noise_summary = finite_noise.groupby(['sigma_y', 'target', 'group']).agg(
        week_mean=('finite_week', 'mean'), week_sd=('finite_week', 'std'),
        delta_mean=('delta', 'mean'), feasible_fraction=('feasible', 'mean'),
        coverage_mean=('coverage_at_base', 'mean')).reset_index()
    return {'31_生存样本与划分': cohort, '31_训练集标准化参数': scaling,
            '31_训练损失': pd.concat(histories, ignore_index=True),
            '31_个体达标概率': pd.DataFrame(predictions), '31_生存模型评估': pd.DataFrame(metrics),
            '32_BMI分组策略': pd.concat(strategies, ignore_index=True),
            '32_GA与动态规划对照': pd.concat(audits, ignore_index=True),
            '32_GA收敛': pd.concat(traces, ignore_index=True), '33_排列重要性': pd.DataFrame(importances),
            '33_读数误差模拟': noise, '33_读数误差汇总': noise_summary}
