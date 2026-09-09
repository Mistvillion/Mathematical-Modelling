"""计算入口调度；每步可独立运行，依赖只在输入指纹一致时复用。"""
import pandas as pd
from .data import load_data, preprocess
from .constants import PAPER_Q2, PAPER_Q3
from .reporting import cached, save_tables, summarize


def paper_comparison(strategies, question):
    rows = []
    if question == 2:
        reference = PAPER_Q2
        chosen = strategies[(strategies.solver == 'GA') & (strategies.k == 3)]
    else:
        reference = PAPER_Q3
        chosen = strategies[(strategies.solver == 'GA') & (strategies['mode'] == 'paper_history')]
    for row in chosen.itertuples():
        if row.target not in reference:
            continue
        ref = reference[row.target]
        paper_week = ref[-1][row.group - 1]
        item = {'target': row.target, 'group': row.group, 'reproduced_week': row.week,
                'paper_week': paper_week, 'week_difference': row.week - paper_week,
                'reproduced_n': row.n, 'reproduced_upper': row.bmi_upper,
                'paper_upper': ref[0][row.group - 1] if row.group < 3 else float('inf')}
        if question == 2:
            item['paper_n'] = ref[1][row.group - 1]
        rows.append(item)
    columns = ['target', 'group', 'reproduced_week', 'paper_week', 'week_difference',
               'reproduced_n', 'reproduced_upper', 'paper_upper'] + (['paper_n'] if question == 2 else [])
    return pd.DataFrame(rows, columns=columns)


def attainment(data, args):
    result = cached('21_个体GPR达标时间', args, 'gpr')
    if result is None:
        run_step('gpr', args, data)
        result = cached('21_个体GPR达标时间', args, 'gpr')
    return result


def run_step(step, args, data=None):
    data = data or load_data(args.data)
    print(f'开始 {step}', flush=True)
    if step == 'preprocess':
        tables = preprocess(data)
    elif step == 'mixed':
        from .mixed import analyse
        tables = analyse(data)
    elif step == 'gpr':
        from .gpr import estimate
        person, curves, logs = estimate(data.male, args.seed, args.gpr_restarts)
        tables = {'21_个体GPR达标时间': person, '21_GPR预测曲线': curves, '21_GPR拟合日志': logs}
    elif step == 'grouping':
        from .optimization import optimise
        from .gpr import measurement_sensitivity
        person = attainment(data, args)
        finite = person[person.finite.astype(bool)]
        strategies, audit, trace = optimise(finite.mean_bmi, finite.t_attain, args.targets, [3, 4, 5, 6], args)
        noise, summary = measurement_sensitivity(data.male, strategies, args.sensitivity_repeats, args.seed)
        tables = {'22_BMI分组策略': strategies, '22_GA与动态规划对照': audit, '22_GA收敛': trace,
                  '22_论文表3至6对照': paper_comparison(strategies, 2),
                  '22_优化排除的删失孕妇': person[~person.finite.astype(bool)],
                  '23_读数误差模拟': noise, '23_读数误差汇总': summary}
    elif step == 'survival':
        from .survival import analyse
        tables = analyse(data, attainment(data, args), args)
        tables['32_论文表8至9对照'] = paper_comparison(tables['32_BMI分组策略'], 3)
    elif step == 'classification':
        from .classification import analyse
        tables = analyse(data, args)
    else:
        raise ValueError(f'未知步骤: {step}')
    save_tables(tables, args, step)
    return tables


def step_main(step):
    from .cli import parse
    args = parse(f'C132复现：{step}')
    run_step(step, args)
    summarize(args)
