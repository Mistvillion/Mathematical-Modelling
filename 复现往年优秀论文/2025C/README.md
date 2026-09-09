# 2025C：C132论文学习与代码复现

本目录复现2025年全国大学生数学建模竞赛C题“NIPT的时点选择与胎儿的异常判定”的C132论文《基于统计建模与元学习的NIPT检测决策优化与异常识别》。沿用2022C的组织方式：计算入口、独立绘图脚本、结果表和学习文档分开存放。

四问均已从题目原始附件运行。当前提供 **48张CSV结果表、35份单图脚本及PNG、1份Excel汇总**，并增加一键运行入口、运行指纹和数值验证。实现依据正文和附录重写，包含明确修复及补充对照，不能把它全部视为作者原始程序，也不宣称第三、四问达到论文声称的精度。

## 1. 从哪里开始

| 目的 | 文件 |
|---|---|
| 逐问理解思路、公式与方法选择 | [C132论文逐题解读与建模方法](docs/C132论文逐题解读与建模方法.md) |
| 核对正文、附录与本实现的差异 | [复现说明](docs/复现说明.md) |
| 学习论文优点与可改进之处 | [论文评价](docs/论文评价.md) |
| 查找原图位置与单图代码 | [图表索引](docs/图表索引.md) |
| 查找每张结果表 | [结果表索引](docs/结果表索引.md) |
| 浏览全部数值 | [计算结果汇总.xlsx](outputs/计算结果汇总.xlsx) |
| 阅读本次运行摘要 | [运行结果](outputs/运行结果.md) |
| 查看已有验证范围 | [验证记录](outputs/验证记录.md) |

原始材料：

- [C132论文PDF](../../准备资料/往年优秀论文/2025年高教社杯全国大学生数学建模竞赛优秀论文/C132.pdf)
- [2025年C题题目](../../准备资料/往年真题/2025年（A、B、C、D、E题）/C题/C题.pdf)
- [题目附件.xlsx](../../准备资料/往年真题/2025年（A、B、C、D、E题）/C题/附件.xlsx)

PDF共65页，部分正文页放在文件末尾，并非缺页：PDF p61—64对应印刷p7—10，PDF p65对应印刷p36。文档均区分两套页码。

## 2. 目录结构

```text
2025C/
├── AGENTS.md
├── README.md
├── requirements.txt
├── scripts_tables/
│   ├── 00_run_all.py                 # 一键计算、汇总、绘图和验证
│   ├── 00_verify.py                  # 验证已有结果
│   ├── 00_preprocess.py
│   ├── 11_mixed_effects.py
│   ├── 21_gpr_attainment.py
│   ├── 22_bmi_optimization.py
│   ├── 31_deep_survival.py
│   ├── 41_stacked_classification.py
│   └── data.py、mixed.py、gpr.py、optimization.py、survival.py等计算模块
├── scripts_figures/                  # 35份单图脚本，每份都有draw()
├── docs/                            # 逐题解读、评价、复现说明、图表索引
└── outputs/
    ├── tables/                      # 48张CSV，UTF-8 BOM
    ├── figures/                     # 35张PNG
    ├── 计算结果汇总.xlsx
    ├── 运行结果.md
    ├── 验证记录.md
    ├── run_metadata.json
    └── validation.json
```

编号按问题和工作步骤组织：`00`为预处理或总体流程，`11`为问题一关联分析，`12/13`为模型及诊断，`21/22/23`为GPR、分组和误差分析，`31/32/33`为生存模型、分组和敏感性，`41/42`为异常模型训练和评价。

## 3. 安装与一键运行

需要Python 3.12或更新版本。在仓库根目录执行：

```bash
cd '复现往年优秀论文/2025C'
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts_tables/00_run_all.py
```

Windows使用 `.venv\Scripts\activate`。macOS上的LightGBM需要OpenMP运行库，若导入时报缺少 `libomp.dylib`，执行 `brew install libomp`。当前环境已完成安装；精确Python和依赖版本记录在 [run_metadata.json](outputs/run_metadata.json)。

常用运行方式：

```bash
# 完整复算；默认覆盖本目录同名结果
python scripts_tables/00_run_all.py

# 核验输入、参数和计算代码指纹后跳过已完成步骤，重新汇总和绘图
python scripts_tables/00_run_all.py --resume

# 只计算表格；也会更新Excel和运行摘要
python scripts_tables/00_run_all.py --tables-only

# 独立运行各问；需要GPR结果时会核验缓存或自动重算
python scripts_tables/11_mixed_effects.py
python scripts_tables/22_bmi_optimization.py
python scripts_tables/31_deep_survival.py
python scripts_tables/41_stacked_classification.py

# 独立修改和重绘一张图，使用已计算且指纹一致的表
python scripts_figures/22_strategy.py
python scripts_figures/42_binary_confusion.py --format both --dpi 300

# 验证整套已有结果
python scripts_tables/00_verify.py
```

所有入口按自身位置定位项目，支持从项目外的工作目录启动。绘图脚本读取 `outputs/tables/` 中的计算值，**不在画单张图时偷偷重训网络**；缺表或附件/表指纹变化时会提示先重新计算。每张图的尺寸、坐标、图例和绘制逻辑都在对应文件中，共用字体和保存逻辑位于 [plotting.py](scripts_tables/plotting.py)。可通过环境变量 `C132_FONT` 指定中文字体文件。

## 4. 参数和计算预算

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--data` | 仓库中的题目附件 | 可指定同结构附件路径 |
| `--output` | `2025C/outputs` | 自定义结果根目录 |
| `--seed` | 42 | 数据划分、GPR、GA、PyTorch、Optuna种子 |
| `--gpr-restarts` | 10 | 与附录B.2相同的GPR优化器重启次数 |
| `--population` / `--generations` | 100 / 50 | GA种群和代数 |
| `--epochs` | 200 | 生存模型最大训练轮数；启用早停 |
| `--targets` | 50 75 90 95 99 | 需要计算的目标达标比例 |
| `--dense` | 关闭 | 遍历50—99全部比例 |
| `--trials` / `--meta-trials` | 12 / 20 | 每个外层训练折内的专家/元模型Optuna次数 |
| `--max-estimators` | 500 | LightGBM调参时的树数上限 |
| `--folds` / `--inner-folds` | 3 / 3 | 按孕妇分组的外层/内层验证折数 |
| `--sensitivity-repeats` | 100 | 误差模拟与孕妇Bootstrap次数 |
| `--format` / `--dpi` | png / 180 | 图片格式与分辨率 |

默认采用正文代表比例及受控的调参预算，便于日常复算；因此默认曲线连接的是5个实际计算点。若要扩大到原附录的50个比例、专家50次和元模型100次搜索、1000棵树上限，可执行：

```bash
python scripts_tables/00_run_all.py --dense --trials 50 --meta-trials 100 \
  --max-estimators 1000 --output ./outputs_extended
```

这会显著增加计算量，仍保留本实现按孕妇分组的验证和必要修正，不等于原附录的全部运行口径。正文GA声称500代、10次独立运行，与附录50代、单次运行不同；当前实现明确采用附录规模的一次GA搜索并附加DP最优性审计。不要将默认结果描述成已经执行过上述扩大预算命令。

## 5. 主要结果与边界

| 问题 | 当前结果 | 如何理解 |
|---|---|---|
| 一 | 固定效应系数与论文表1接近；ICC≈0.742925；LR≈752.268；GLS Wald≈440.897 | 随机截距有必要，孕周在观测范围内呈加速正效应，BMI和身高系数为负 |
| 二 | 267人中260人得到有限达标时间、7人仍未达标；默认5个比例的三组GA风险均等于DP最优风险 | 采用原附录GPR规则；分界点存在多个风险等价解，不能只看边界数字是否相同 |
| 三 | 完整轨迹模式测试C-index≈0.893；静态区间删失对照≈0.685 | 两者标签和可用信息不同；完整轨迹模式不是首次检测前的预测精度 |
| 四 | 外层OOF二分类准确率≈88.93%，AUC≈0.670；异常召回率≈4.48% | 准确率与“全部预测无异常”基线相同，不能据此说检测可靠；未复现原文99.7% |

第二、三问的 `target` 是经验达标比例或模型达标概率，不能等同于非整倍体诊断准确率。第二问按附录排除7位无穷时间的孕妇进行优化，第三问则保留删失信息。策略中的 `feasible` 只表示在模型10—40周计算域内达到目标，`within_10_25_weeks` 单独标记是否落在题目常用检测窗口；超过25周的输出属于模型结果和外推限制。

图形覆盖原文图1—21的可复算内容、拆分子图及补充对照，共35张。数字、顺序和统计口径均以运行CSV为准，不为匹配原图曲线而修改计算结果。详细复现边界见 [复现说明](docs/复现说明.md)。
