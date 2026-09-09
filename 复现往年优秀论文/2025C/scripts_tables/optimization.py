"""连续BMI区间的GA搜索，以及同一目标的动态规划审计。

排序后每组是连续区间，目标对区间可加；因而可用O(k*n²)动态规划求全局最优。
GA结果与DP结果分别输出，绝不把DP最优解当成GA运行结果。
"""
from bisect import insort
import numpy as np
import pandas as pd


def daily_quantile(cdf, weeks, target):
    """CDF未达到目标返回inf；达到时向上取整到天，禁止截顶冒充达标。"""
    hit = np.flatnonzero(np.asarray(cdf) >= target)
    if not len(hit):
        return np.inf
    j = hit[0]
    if j == 0:
        return float(weeks[0])
    fraction = (target - cdf[j - 1]) / (cdf[j] - cdf[j - 1])
    continuous = weeks[j - 1] + fraction * (weeks[j] - weeks[j - 1])
    return float(np.ceil(continuous * 7 - 1e-10) / 7)


def risk(week, question=2):
    week = np.asarray(week)
    if question == 2:
        return np.where(week <= 12, 1., np.where(week <= 27, 10., 100.))
    # 附录B.3使用1/5/10及28周；与问题二1/10/100及27周分开。
    return np.where(week <= 12, 1., np.where(week <= 28, 5., 10.))


def segment_weeks(values, target, kind, weeks=None):
    n = len(values)
    result = np.full((n + 1, n + 1), np.inf)
    if kind == 'gpr':
        for start in range(n):
            ordered = []
            for end in range(start + 1, n + 1):
                insort(ordered, float(values[end - 1]))
                h = (len(ordered) - 1) * target
                low = int(np.floor(h))
                high = int(np.ceil(h))
                result[start, end] = ordered[low] + (h - low) * (ordered[high] - ordered[low])
    else:
        cumulative = np.vstack([np.zeros(values.shape[1]), np.cumsum(values, axis=0)])
        for start in range(n):
            group_cdf = (cumulative[start + 1:] - cumulative[start]) / np.arange(1, n - start + 1)[:, None]
            for j, cdf in enumerate(group_cdf, start + 1):
                result[start, j] = daily_quantile(cdf, weeks, target)
    return result


def partition_costs(bmi, segment, k, question):
    n = len(bmi)
    minimum = int(np.floor(n / k * 0.6)) if question == 2 else 50
    sizes = np.arange(n + 1)[None, :] - np.arange(n + 1)[:, None]
    valid_cut = np.r_[True, np.diff(bmi) > 0, True]
    valid = (sizes >= minimum) & valid_cut[:, None] & valid_cut[None, :]
    costs = np.where(valid, sizes * np.where(np.isfinite(segment), risk(segment, question), 1e6), np.inf)
    secondary = np.where(valid, sizes * np.where(np.isfinite(segment), segment, 1e4), np.inf)
    return costs, secondary, minimum, valid_cut


def exact_partition(costs, secondary, k):
    n = len(costs) - 1
    dp = np.full((k + 1, n + 1), np.inf)
    tie = np.full_like(dp, np.inf)
    back = np.full_like(dp, -1, dtype=int)
    dp[0, 0] = tie[0, 0] = 0.
    for group in range(1, k + 1):
        for end in range(1, n + 1):
            totals = dp[group - 1, :end] + costs[:end, end]
            seconds = tie[group - 1, :end] + secondary[:end, end]
            ix = np.lexsort((np.arange(end), seconds, totals))[0]
            if np.isfinite(totals[ix]):
                dp[group, end], tie[group, end], back[group, end] = totals[ix], seconds[ix], ix
    if not np.isfinite(dp[k, n]):
        raise ValueError(f'无法形成{k}个符合最小组规模且不拆分相同BMI的组')
    boundaries, end = [n], n
    for group in range(k, 0, -1):
        end = back[group, end]
        boundaries.append(int(end))
    return np.array(boundaries[::-1][1:-1]), dp[k, n]


def genetic_partition(costs, valid_cut, k, population, generations, seed):
    """附录B.2型均匀交叉、随机BMI变异、锦标赛及最佳历史个体记录。"""
    rng = np.random.default_rng(seed)
    n = len(costs) - 1
    allowed = np.flatnonzero(valid_cut[1:-1]) + 1
    def draw():
        return np.sort(rng.choice(allowed, k - 1, replace=False))
    def score(cuts):
        bounds = np.r_[0, np.sort(cuts), n]
        val = costs[bounds[:-1], bounds[1:]]
        return float(np.sum(val)) if np.isfinite(val).all() else 1e12
    pop = [draw() for _ in range(population)]
    best, best_value, history = None, np.inf, []
    for generation in range(generations + 1):
        fitness = np.array([score(c) for c in pop])
        winner = int(np.argmin(fitness))
        if fitness[winner] < best_value:
            best, best_value = pop[winner].copy(), fitness[winner]
        history.append({'generation': generation, 'best_risk': best_value,
                        'feasible_fraction': np.mean(fitness < 1e12)})
        if generation == generations:
            break
        selected = []
        for _ in range(population):
            competitors = rng.integers(0, population, 3)
            selected.append(pop[competitors[np.argmin(fitness[competitors])]].copy())
        for j in range(0, population - 1, 2):
            if rng.random() < 0.9:
                mask = rng.random(k - 1) < 0.5
                left, right = selected[j].copy(), selected[j + 1].copy()
                selected[j][mask], selected[j + 1][mask] = right[mask], left[mask]
        for individual in selected:
            if rng.random() < 0.3:
                mask = rng.random(k - 1) < 0.2
                individual[mask] = rng.choice(allowed, int(mask.sum()))
            individual.sort()
        pop = selected
    if best_value >= 1e12:
        raise RuntimeError('GA没有找到可行分组，请增加种群或代数')
    return best, best_value, pd.DataFrame(history)


def optimise(bmi, values, targets, ks, args, kind='gpr', weeks=None, mode='GPR'):
    order = np.argsort(bmi, kind='stable')
    bmi, values = np.asarray(bmi)[order], np.asarray(values)[order]
    question = 2 if kind == 'gpr' else 3
    strategies, audits, traces = [], [], []
    for target in targets:
        segment = segment_weeks(values, target / 100, kind, weeks)
        for k in ks:
            costs, secondary, minimum, valid_cut = partition_costs(bmi, segment, k, question)
            exact, optimum = exact_partition(costs, secondary, k)
            ga, ga_risk, trace = genetic_partition(costs, valid_cut, k, args.population,
                                                  args.generations, args.seed + int(target) * 11 + k)
            trace = trace.assign(target=target, k=k, mode=mode)
            traces.append(trace)
            audits.append({'mode': mode, 'target': target, 'k': k, 'ga_risk': ga_risk,
                           'dp_risk': optimum, 'gap': ga_risk - optimum,
                           'minimum_group_size': minimum, 'n_subjects': len(bmi)})
            for solver, cuts in [('GA', ga), ('DP', exact)]:
                bounds = np.r_[0, cuts, len(bmi)]
                edges = np.r_[-np.inf, bmi[cuts - 1], np.inf]
                for group, (start, end) in enumerate(zip(bounds[:-1], bounds[1:]), 1):
                    week = segment[start, end]
                    row = {'mode': mode, 'solver': solver, 'target': target, 'k': k, 'group': group,
                           'bmi_lower': edges[group - 1], 'bmi_upper': edges[group], 'n': end - start,
                           'week': week, 'feasible': bool(np.isfinite(week)),
                           'within_10_25_weeks': bool(10 <= week <= 25),
                           'group_risk': costs[start, end], 'minimum_group_size': minimum}
                    if kind == 'gpr':
                        part = values[start:end]
                        row['empirical_coverage'] = np.mean(part <= week)
                        row['empirical_inverse_week'] = np.sort(part)[int(np.ceil(len(part) * target / 100)) - 1]
                    else:
                        mean_cdf = values[start:end].mean(axis=0)
                        row['predicted_coverage'] = np.interp(week, weeks, mean_cdf) if np.isfinite(week) else mean_cdf[-1]
                    strategies.append(row)
    return pd.DataFrame(strategies), pd.DataFrame(audits), pd.concat(traces, ignore_index=True)
