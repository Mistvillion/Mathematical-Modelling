# 练习2022C：C155 论文学习与代码复现

本目录围绕2022年全国大学生数学建模竞赛C题“古代玻璃制品的成分分析与鉴别”，整理C155论文的解题思路、Python复现代码和计算结果。代码覆盖数据预处理、四问分析、敏感性检验及逐图绘制；具体方法与原文差异在 `docs/` 中说明。

当前文件按用途分为：**`scripts_tables/` 存放解题入口和计算模块，`scripts_figures/` 存放单图脚本，`outputs/` 存放结果，`docs/` 存放学习文档。** 复现代码包含根据正文补写和修正的内容，不能将其全部视为作者公开的原始程序。

## 1. 从哪里开始

| 目的 | 对应文件 |
|---|---|
| 理解每一问的思路、方法选择与模型建立 | [C155论文逐题解读与建模方法](docs/C155论文逐题解读与建模方法.md) |
| 了解论文优点、漏洞和学习建议 | [论文评价](docs/论文评价.md) |
| 核对数据口径、参数、原文矛盾与实现修正 | [复现说明](docs/复现说明.md) |
| 查询图名、原文页码和绘图脚本名称 | [图表索引](docs/图表索引.md) |
| 直接查看计算表格 | [计算结果汇总.xlsx](outputs/计算结果汇总.xlsx) |
| 查看已保存的主要结果 | [运行结果](outputs/运行结果.md) |
| 查看已有验证范围与限制 | [验证记录](outputs/验证记录.md) |

题目与论文位于仓库根目录的 `准备资料/` 中：

- [论文原文 C155.pdf](../准备资料/往年优秀论文/2022年高教社杯全国大学生数学建模竞赛优秀论文/C155.pdf)
- [2022年C题题目](../准备资料/往年真题/2022年（A、B、C、D、E题）/C题/C题.pdf)
- [题目原始附件.xlsx](../准备资料/往年真题/2022年（A、B、C、D、E题）/C题/附件.xlsx)

## 2. 当前目录结构

```text
练习2022C/
├── README.md
├── requirements.txt
├── docs/
│   ├── C155论文逐题解读与建模方法.md
│   ├── 论文评价.md
│   ├── 复现说明.md
│   └── 图表索引.md
├── scripts_tables/                    # 8个解题入口及13个辅助/包文件
│   ├── 00_preprocess.py
│   ├── 11_association.py
│   ├── 12_descriptive.py
│   ├── 13_weathering_prediction.py
│   ├── 21_decision_trees.py
│   ├── 22_subclasses.py
│   ├── 31_identify_unknown.py
│   ├── 41_grey_relations.py
│   ├── data.py                        # 附件读取、数据校验、CSV保存
│   ├── composition.py                 # 闭合、对数比、零值与逆变换
│   ├── association.py                 # 列联检验
│   ├── descriptive.py                 # 描述统计
│   ├── weathering.py                  # 风化分组、回归与预测
│   ├── classification.py              # 决策树、未知鉴别与敏感性
│   ├── subclasses.py                  # R/Q型聚类与亚类敏感性
│   ├── grey.py                        # 灰色关联分析
│   ├── constants.py                   # 路径、成分顺序、论文参考值
│   ├── cli.py                         # 命令行参数
│   ├── plotting.py                    # 共用绘图样式、字体与保存
│   ├── reporting.py                   # Excel汇总和运行记录函数
│   └── __init__.py
├── scripts_figures/                   # 91个单图Python脚本
│   ├── fig01a_boxplot.py
│   ├── fig02_potassium_ward.py
│   ├── fig04_paper_decision_tree.py
│   ├── fit_potassium_SiO2.py
│   ├── appendixA_lead_barium_mixed_cluster.py
│   └── …
└── outputs/
    ├── tables/                        # 41个CSV结果文件
    ├── figures/                       # 91张PNG图片
    ├── 计算结果汇总.xlsx
    ├── 运行结果.md
    ├── 验证记录.md
    └── run_metadata.json              # 附件哈希、运行参数和依赖版本记录
```

绘图脚本可从 [图表索引](docs/图表索引.md) 直接打开；历史验证范围见 [验证记录](outputs/验证记录.md)。

## 3. 解题代码导航

`scripts_tables/` 中，带数字前缀的8个文件是命令行入口；同目录中不带数字前缀的文件提供计算函数。`11` 表示问题1第一小问，`13` 表示问题1第三小问，`00` 表示预处理。计算模块内部使用相对导入，应作为 `scripts_tables` 包的一部分使用。

| 步骤 | 入口脚本 | 主要实现 | 结果文件前缀 |
|---|---|---|---|
| 数据读取、有效性筛选、闭合与对数比变换 | [00_preprocess.py](scripts_tables/00_preprocess.py) | [data.py](scripts_tables/data.py)、[composition.py](scripts_tables/composition.py) | `00_` |
| 问题1.1：风化与类型、纹饰、颜色的关系 | [11_association.py](scripts_tables/11_association.py) | [association.py](scripts_tables/association.py) | `11_` |
| 问题1.2：四组化学成分描述统计 | [12_descriptive.py](scripts_tables/12_descriptive.py) | [descriptive.py](scripts_tables/descriptive.py) | `12_` |
| 问题1.3：Ward聚类、阶段回归与风化前预测 | [13_weathering_prediction.py](scripts_tables/13_weathering_prediction.py) | [weathering.py](scripts_tables/weathering.py) | `13_` |
| 问题2.1：决策树规则、训练与分组验证 | [21_decision_trees.py](scripts_tables/21_decision_trees.py) | [classification.py](scripts_tables/classification.py) | `21_` |
| 问题2.2：R/Q型亚类划分及敏感性 | [22_subclasses.py](scripts_tables/22_subclasses.py) | [subclasses.py](scripts_tables/subclasses.py) | `22_`、`23_` |
| 问题3：未知样本鉴别及敏感性 | [31_identify_unknown.py](scripts_tables/31_identify_unknown.py) | [classification.py](scripts_tables/classification.py) | `31_`、`32_` |
| 问题4：灰色关联及类型间差异 | [41_grey_relations.py](scripts_tables/41_grey_relations.py) | [grey.py](scripts_tables/grey.py) | `41_`、`42_` |

每个入口从原始附件读取所需数据，并调用计算模块生成结果表，不依赖前一个入口产生的CSV缓存。`reporting.py` 提供汇总函数，但目前没有统一调度脚本；单独运行某一步不会自动更新Excel汇总和整套运行记录。

## 4. 绘图代码导航

`scripts_figures/` 按“一张图一份代码”拆分，每个文件都有自己的 `draw(data, args)`。修改单张图的尺寸、颜色、坐标轴和图例时，直接编辑对应文件；共用样式和保存逻辑位于 [plotting.py](scripts_tables/plotting.py)。

| 文件命名 | 数量 | 内容 | 示例 |
|---|---:|---|---|
| `fig01`—`fig09` | 19 | 正文图1—9及拆开的子图 | [fig01a_boxplot.py](scripts_figures/fig01a_boxplot.py)、[fig06c_correlation.py](scripts_figures/fig06c_correlation.py) |
| `appendixA_*` | 2 | 原始与预测数据混合聚类 | [appendixA_lead_barium_mixed_cluster.py](scripts_figures/appendixA_lead_barium_mixed_cluster.py) |
| `appendixB_*` | 4 | 两类玻璃各分支的二次聚类 | [appendixB_potassium_branch1.py](scripts_figures/appendixB_potassium_branch1.py) |
| `fit_*` | 28 | 两类玻璃各14种成分的回归拟合 | [fit_potassium_SiO2.py](scripts_figures/fit_potassium_SiO2.py) |
| `supplement_grey_*` | 26 | 除SiO₂外其余母序列的灰色关联曲线 | [supplement_grey_lead_barium_PbO.py](scripts_figures/supplement_grey_lead_barium_PbO.py) |
| `supplement_R_*` | 4 | 四组R型变量聚类树 | [supplement_R_a.py](scripts_figures/supplement_R_a.py) |
| `supplement_Q_*` | 4 | 四组Q型亚类聚类树 | [supplement_Q_a.py](scripts_figures/supplement_Q_a.py) |
| `supplement_tree_*` | 4 | 全成分与预选PbO的重训决策树 | [supplement_tree_04_pbo.py](scripts_figures/supplement_tree_04_pbo.py) |

文件名中的 `potassium` 表示高钾，`lead_barium` 表示铅钡。输出图片通常与脚本同名，保存在 `outputs/figures/`。91张图包含重绘、附录方法生成的图和补充对照，并非PDF原有91张图；来源与性质见各脚本开头的说明及 [图表索引](docs/图表索引.md)。

## 5. 环境、输入与运行方式

### 安装依赖

需要Python 3.11或更新版本。以下命令从仓库根目录进入 `练习2022C` 并安装依赖：

```bash
cd '练习2022C'
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows命令提示符下使用 `.venv\Scripts\activate` 激活环境。当前依赖范围见 [requirements.txt](requirements.txt)；已有结果生成时的依赖版本记录在 [run_metadata.json](outputs/run_metadata.json) 中。

默认附件路径由 `scripts_tables/constants.py` 按文件位置定位，相对于 `练习2022C` 目录为：

```text
../准备资料/往年真题/2022年（A、B、C、D、E题）/C题/附件.xlsx
```

附件应保留 `表单1`、`表单2`、`表单3` 的名称和原始化学成分列。读取器会检查列结构、文物编号匹配、含量有效性等条件。

### 独立运行

解题入口和绘图脚本统一从 `scripts_tables` 包导入计算模块。入口会根据自身文件位置定位 `练习2022C`，可从其他工作目录通过脚本路径启动；默认附件和默认输出目录也按项目位置定位。

导入修复后，8个解题入口和91个绘图入口已分别从项目外的临时目录执行成功，生成的41个CSV和91张PNG与已有结果逐文件一致。验证输出写入临时目录，原有 `outputs/` 文件保持不变。

在 `练习2022C` 目录下，可以直接运行以下命令：

```bash
# 问题1.1：生成关联检验结果表
python scripts_tables/11_association.py

# 问题1.3：生成阶段回归与风化前预测
python scripts_tables/13_weathering_prediction.py

# 单独生成图1a
python scripts_figures/fig01a_boxplot.py

# 指定原始附件、输出位置和图像格式
python scripts_figures/fit_potassium_SiO2.py --data '/你的目录/附件.xlsx' --output './新结果' --format both --dpi 300
```

也可以直接在Python中使用计算模块。例如，在本目录检查有效采样点数量：

```bash
python -c "from scripts_tables.data import load_data; print(len(load_data().samples))"
```

使用原附件时应输出 `67`。

当前命令行参数定义在 [cli.py](scripts_tables/cli.py)：

| 参数 | 默认值 | 用途 |
|---|---|---|
| `--data` | 仓库 `准备资料/往年真题/` 下的2022年C题 `附件.xlsx` | 指定原始附件 |
| `--output` | `练习2022C/outputs` | 指定结果根目录；表和图分别写入其 `tables/`、`figures/` |
| `--seed` | `2022` | 有随机过程的分析所用种子 |
| `--permutation-repeats` | `20000` | 稀疏列联表Monte Carlo次数 |
| `--sensitivity-repeats` | `200` | 每个敏感性场景的重复次数 |
| `--format` | `png` | 绘图格式，可选 `png`、`pdf`、`both` |
| `--dpi` | `180` | 图片输出分辨率 |

各脚本只使用与本步骤相关的参数。输出目录中的同名文件会被覆盖。绘图采用无界面的Agg后端；中文字体自动检测，也可通过环境变量 `C155_FONT` 指定字体文件。

## 6. 已有结果与复现边界

当前 `outputs/` 保存41个CSV、91张PNG和1份Excel汇总。运行记录对应58件文物、67个有效采样点（高钾18、铅钡49），剔除15、17；两种风化前预测方法各输出38行。

| 要查看的结果 | 文件 |
|---|---|
| 清洗后的采样点和三种风化标签 | [00_有效采样点_原始含量.csv](outputs/tables/00_有效采样点_原始含量.csv) |
| 风化与类型、纹饰、颜色的关系 | [11_独立性检验.csv](outputs/tables/11_独立性检验.csv) |
| 四组描述统计 | [12_四组描述统计.csv](outputs/tables/12_四组描述统计.csv) |
| 四阶段中心、回归系数与R² | [13_回归系数与四阶段中心.csv](outputs/tables/13_回归系数与四阶段中心.csv) |
| 风化前预测及方法对照 | [signed_sqrt预测](outputs/tables/13_风化前预测_signed_sqrt.csv)、[shift预测](outputs/tables/13_风化前预测_shift.csv) |
| 决策树训练与验证 | [21_决策树评估.csv](outputs/tables/21_决策树评估.csv) |
| 亚类划分与敏感性 | [22_Q亚类成员.csv](outputs/tables/22_Q亚类成员.csv)、[23_亚类敏感性.csv](outputs/tables/23_亚类敏感性.csv) |
| A1—A8鉴别与边界距离 | [31_未知样本分类与边界距离.csv](outputs/tables/31_未知样本分类与边界距离.csv) |
| 灰色关联与原论文数值差异 | [42_论文表28对照.csv](outputs/tables/42_论文表28对照.csv) |

理解结果时需要注意：

- 文物表面风化、采样点风化和论文聚类推定风化是不同标签，不能混用。
- 高钾表11与表13的第3、4组顺序相反；实现保留两套回归顺序的对照，主拟合按表13/14。
- 论文固定PbO规则、预选PbO重训树和全14成分重训树分别记录，不能把不同模型的结果当成同一输出。
- 零掩码、稀疏表检验和预测根式包含明确修正；表17/18预测、部分亚类成员及表28灰色关联度不宣称逐值复现。
- 风化前没有配对真值可供验证，预测满足非负与总和100%只说明数值约束成立，不证明恢复准确。

算法细节和适用范围见 [复现说明](docs/复现说明.md)。
