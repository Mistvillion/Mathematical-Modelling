"""可重复执行的数值契约验证：独立小例、数据一致性、删失、概率和分组隔离。"""
import ast
from itertools import combinations
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
import torch
from .constants import PROJECT
from .data import load_data, parse_week
from .optimization import daily_quantile, segment_weeks, partition_costs, exact_partition
from .reporting import sha256
from .survival import HybridSurvivalNet, loss_function


def validate(args):
    records = []
    def record(name, function):
        try:
            detail = function()
            records.append({'check': name, 'passed': True, 'detail': str(detail or '通过')})
        except Exception as exc:
            records.append({'check': name, 'passed': False, 'detail': f'{type(exc).__name__}: {exc}'})
    def require(condition, message):
        if not condition:
            raise AssertionError(message)
    def table(name):
        return pd.read_csv(args.output / 'tables' / f'{name}.csv')
    data = load_data(args.data)

    def parsing():
        require(parse_week('16W+1') == 16 + 1 / 7 and parse_week('16w') == 16, '孕周周天转换错误')
        for invalid in ['16w+7', 'bad', float('nan')]:
            try:
                parse_week(invalid)
            except ValueError:
                continue
            raise AssertionError(f'坏输入未被拒绝: {invalid}')
    record('孕周转换及坏输入', parsing)

    def rows():
        require(len(data.male) == len(table('00_男胎清洗数据')), '男胎记录数变化')
        require(len(data.female) == len(table('00_女胎清洗数据')), '女胎记录数变化')
        require(data.male.person_id.nunique() == len(table('00_男胎孕妇汇总')), '孕妇聚合不是一人一行')
        require(data.female.state.value_counts().to_dict() == table('00_女胎标签分布').set_index('state').records.to_dict(), '女胎AB标签分布变化')
        return f'男胎{len(data.male)}条/{data.male.person_id.nunique()}人；女胎{len(data.female)}条/{data.female.person_id.nunique()}人'
    record('原始附件与清洗、标签一致', rows)

    def gpr():
        result = table('21_个体GPR达标时间')
        require(result.person_id.is_unique and len(result) == data.male.person_id.nunique(), 'GPR孕妇行重复或丢失')
        require(np.array_equal(np.isfinite(result.t_attain), result.finite), '无穷值标记不一致')
        require((result.t_attain >= 10).all(), '违反10周下界')
        require((result.t_attain <= result.observed_first + 1e-9).all(), '真实达标点未修正GPR')
        return f'有限{result.finite.sum()}人；未达标{(~result.finite).sum()}人'
    record('GPR三阶段规则和删失保留', gpr)

    def paper_reference():
        from .constants import DATA
        if sha256(args.data) != sha256(DATA):
            return '自定义附件：不强制对齐原论文数值'
        coefficients = table('12_固定效应与论文表1对照')
        require(coefficients.difference.abs().max() < 1e-5, '第一问系数偏离论文参考值')
        diagnostics = table('13_模型检验').set_index('metric').value
        require(abs(diagnostics['ICC_REML'] - .743) < .0005, 'ICC不匹配原文')
        require(abs(diagnostics['Wald_fixed_4df'] - 440.90) < .01, 'GLS Wald不匹配原文')
        require(abs(diagnostics['LR_ML_mixture_chi2'] - 752.27) < .01, 'LR不匹配原文')
        return '表1系数误差<1e-5；ICC、LR与GLS Wald对齐原文精度'
    record('第一问独立论文参考值核对', paper_reference)

    def optimality():
        bmi = np.array([22., 24., 26., 28., 30., 32., 34., 36.])
        times = np.array([10., 12., 15., 11., 18., 13., 25., 29.])
        segments = segment_weeks(times, .75, 'gpr')
        costs, tie, _, _ = partition_costs(bmi, segments, 3, 2)
        _, value = exact_partition(costs, tie, 3)
        brute = min(sum(costs[a, b] for a, b in zip((0,) + cuts, cuts + (len(bmi),))) for cuts in combinations(range(1, len(bmi)), 2))
        require(value == brute, f'DP={value}, 穷举={brute}')
        duplicate_bmi = bmi.copy(); duplicate_bmi[3] = duplicate_bmi[2]
        _, _, _, cuts = partition_costs(duplicate_bmi, segments, 3, 2)
        require(not cuts[3], '相同BMI被允许拆成两个组')
        return f'8人例：DP风险={value}，与所有分割点穷举相等'
    record('动态规划对独立穷举例及重复BMI边界', optimality)

    def quantile():
        cdf = np.array([.1, .4, .8]); weeks = np.array([10., 11., 12.])
        answer = daily_quantile(cdf, weeks, .5)
        require(abs(answer - (11 + 2 / 7)) < 1e-10, '日插值未向上取整')
        require(np.isinf(daily_quantile(cdf, weeks, .9)), '未达标被截顶为末周')
    record('日插值、尾部未达标处理', quantile)

    def padding():
        torch.manual_seed(123)
        model = HybridSurvivalNet().eval()
        static = torch.randn(2, 4)
        short = torch.randn(2, 3, 2)
        long = torch.cat([short, torch.randn(2, 5, 2) * 100], dim=1)
        lengths = torch.tensor([2, 3])
        with torch.no_grad():
            a, b = model(static, short, lengths), model(static, long, lengths)
            require(torch.allclose(a, b, atol=1e-7), '填充值影响LSTM结果')
            empty = torch.zeros(2, dtype=torch.long)
            require(torch.allclose(model(static, short, empty), model(static, long, empty), atol=1e-7), '静态对照偷用了序列')
    record('LSTM填充不变性与静态模式隔离', padding)

    def censoring():
        # 三格概率0.2/0.3/0.5，末格是尾部。
        logits = torch.log(torch.tensor([[.2, .3, .5]]))
        right = loss_function(logits, torch.tensor([0]), torch.tensor([1]), torch.tensor([False]), interval=True)
        interval = loss_function(logits, torch.tensor([0]), torch.tensor([1]), torch.tensor([True]), interval=True)
        require(np.isclose(right.item(), -np.log(.5)), '右删失似然错误')
        require(np.isclose(interval.item(), -np.log(.3)), '区间删失似然错误')
    record('删失似然的手算概率例', censoring)

    def probabilities():
        d = table('31_个体达标概率')
        for _, g in d.groupby(['mode', 'person_id']):
            require(np.all(np.diff(g.cdf) >= -2e-6), 'CDF不单调')
            require(g.cdf.between(-2e-6, 1+2e-6).all(), 'CDF超出0..1')
            require(abs(g.pmf.sum() + g.tail_after40.iloc[0] - 1) < 2e-6, 'PMF及尾部之和不为1')
        cohort = table('31_生存样本与划分')
        require(cohort.person_id.is_unique, '同一孕妇跨生存训练/验证/测试集')
        return str(cohort.split.value_counts().to_dict())
    record('生存概率守恒、单调性及孕妇划分', probabilities)

    def grouping():
        for name, question in [('22_BMI分组策略', 2), ('32_BMI分组策略', 3)]:
            d = table(name)
            for _, g in d.groupby(['mode', 'solver', 'target', 'k']):
                g = g.sort_values('group')
                require((g.n >= g.minimum_group_size).all(), '存在过小分组')
                require(np.allclose(g.bmi_upper.to_numpy()[:-1], g.bmi_lower.to_numpy()[1:]), '分组边界有间隙或重叠')
                require(np.isneginf(g.bmi_lower.iloc[0]) and np.isposinf(g.bmi_upper.iloc[-1]), '边界未覆盖全部BMI')
                require(g.n.sum() == (table('21_个体GPR达标时间').finite.sum() if question == 2 else data.male.person_id.nunique()), '分组人数不守恒')
                if question == 3:
                    require((g.loc[g.feasible, 'predicted_coverage'] >= g.loc[g.feasible, 'target']/100 - 2e-6).all(), '可行方案未达到目标概率')
        for name in ['22_GA与动态规划对照', '32_GA与动态规划对照']:
            require((table(name).gap >= -1e-6).all(), 'GA风险低于同目标全局最优，存在错误')
    record('BMI分组覆盖、规模约束和风险审计', grouping)

    def classification():
        from .classification import FEATURES, META_RAW
        prohibited = {'person_id', 'row_id', 'state', 'abnormal', 'abnormal_text', 'birth_unhealthy', 'health_text'}
        require(not prohibited.intersection(FEATURES + META_RAW), '特征含标签/身份/未来结局')
        split = table('41_孕妇分组验证划分')
        for _, g in split.groupby('outer_fold'):
            require(set(g[g.role=='train'].person_id).isdisjoint(set(g[g.role=='test'].person_id)), '外层训练验证孕妇重合')
        pred = table('42_女胎逐记录OOF预测')
        require(pred.row_id.is_unique and len(pred)==len(data.female), 'OOF记录重复或遗漏')
        require((pred.groupby('person_id').fold.nunique()==1).all(), '孕妇跨验证折')
        truth = data.female.set_index('row_id').state.loc[pred.row_id].to_numpy()
        require(np.array_equal(truth, pred.true_state), '预测标签与原始AB不一致')
        p = pred[[f'p_state{i}' for i in range(8)]].to_numpy()
        require(np.allclose(p.sum(axis=1), 1, atol=1e-8), '多分类概率和不为1')
        require(np.all((p>=0)&(p<=1)), '分类概率越界')
        cm = table('42_混淆矩阵')
        for _, g in cm.groupby(['variant', 'kind']):
            require(g['count'].sum()==len(pred), '混淆矩阵人数不守恒')
        return f'{len(pred)}条OOF预测覆盖{pred.person_id.nunique()}位孕妇'
    record('女胎无身份泄漏、按孕妇OOF和概率一致性', classification)

    def files():
        metadata = json.loads((args.output/'run_metadata.json').read_text())
        for step, entry in metadata['steps'].items():
            require(entry['data_sha256']==sha256(args.data), f'{step}使用了不同附件')
            for name, expected in entry['calculation_sha256'].items():
                require(sha256(PROJECT/'scripts_tables'/name)==expected, f'{step}计算代码已变化，请重算')
            for name, expected in entry['table_sha256'].items():
                require(sha256(args.output/'tables'/f'{name}.csv')==expected, f'{name}CSV被更改')
        require(len({json.dumps(x['config'],sort_keys=True) for x in metadata['steps'].values()})==1, '各问参数不同，须检查或统一重算')
        expected_figures=[]
        for script in sorted((PROJECT/'scripts_figures').glob('[0-9]*.py')):
            tree=ast.parse(script.read_text())
            name=next(node.value.value for node in tree.body if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='NAME' for t in node.targets))
            expected_figures.append(name)
            for extension in ['png','pdf'] if args.format=='both' else [args.format]:
                image=args.output/'figures'/f'{name}.{extension}'
                require(image.exists() and image.stat().st_size>1000, f'缺图或空图：{image.name}')
        require(len(expected_figures)==len(set(expected_figures)), '图名重复覆盖')
        return f'{sum(len(x["tables"]) for x in metadata["steps"].values())}张表，{len(expected_figures)}份单图脚本'
    record('数据、计算源代码、CSV指纹和图文件齐全', files)

    def links():
        count = 0
        for path in [PROJECT/'README.md', *sorted((PROJECT/'docs').glob('*.md'))]:
            require(path.exists(), f'缺文档：{path.name}')
            for target in re.findall(r'\]\(([^)]+)\)',path.read_text()):
                target=target.split('#')[0].strip('<>')
                if not target or '://' in target:
                    continue
                destination = (path.parent/target).resolve()
                # 本函数在全部检查结束后写报告；首次运行时该自链接尚未落盘。
                if destination != (args.output/'验证记录.md').resolve():
                    require(destination.exists(), f'{path.name}中的失效链接：{target}')
                count += 1
        return f'检查{count}个本地文档链接'
    record('README及学习文档链接', links)

    passed = all(row['passed'] for row in records)
    (args.output/'validation.json').write_text(json.dumps({'passed': passed, 'checks': records},ensure_ascii=False,indent=2)+'\n')
    lines=['# 验证记录','','由 `scripts_tables/00_verify.py` 执行数值与文件契约检查。',
           '图像排版另经缩略图总览与代表图原尺寸人工检查；统计有效性不由文件检查自动保证。','',
           pd.DataFrame(records).to_markdown(index=False),'',
           '限制：未做外部临床验证；GPR时间不是配对真值；完整轨迹生存模式有目标信息进入输入；少数复合异常仅2条。',
           '单独运行某一步会更新该步CSV及Excel汇总；整套文档、图片与验证记录应通过00_run_all.py重新生成。','']
    (args.output/'验证记录.md').write_text('\n'.join(lines))
    print(f'验证：{sum(row["passed"] for row in records)}/{len(records)}通过',flush=True)
    if not passed:
        raise AssertionError('; '.join(row['check']+': '+row['detail'] for row in records if not row['passed']))
    return records
