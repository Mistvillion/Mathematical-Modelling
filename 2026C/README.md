# 2026 年数学建模 C 题：微网与外部电网电力调控策略

本目录保存 C 题的题面、原始附件、建模文档及 Python 代码。目前已整理问题一的建模思路，并实现附件 1 的日内数据绘图。

## 目录

```text
2026C/
├── CUMCM2026Problems/          # 题目及原始附件
├── scripts_figures/           # 绘图代码
│   └── 1_time_series.py
├── outputs/
│   └── figures/               # 图片：PNG 和 SVG
├── docs/
│   ├── Question_1.md          # 问题一的建模思路
│   └── symbols.md             # 符号说明
├── requirements.txt           # Python 第三方依赖
└── README.md
```

后续表格脚本放入 `scripts_tables/`，表格输出放入 `outputs/tables/`。文件名前缀 `1_` 表示问题一，`0_` 表示预处理。

## 运行问题一绘图

使用 conda 虚拟环境 `2026C`。在本目录下执行：

```bash
conda activate 2026C
python -m pip install -r requirements.txt
python scripts_figures/1_time_series.py
```

也可在仓库总目录下执行：

```bash
conda run --no-capture-output -n 2026C python 2026C/scripts_figures/1_time_series.py
```

脚本根据自身位置定位附件和输出目录，因此不依赖启动时的工作目录。中文字体自动从系统中选择，支持 Noto Sans CJK SC、思源黑体、微软雅黑、黑体、苹方等；如缺少中文字体，安装其中一种后再运行。

## 图与数据口径

数据来自 [附件 1](CUMCM2026Problems/C题/附件/附件1.xlsx) 的 `Sheet1`，共 144 条记录。

| 输出图名 | 横轴 | 纵轴与曲线 |
| --- | --- | --- |
| `1_电价时间折线图` | 时间 | 电价，单位：元/kWh |
| `1_小区负载与光伏发电预测功率时间折线图` | 时间 | 小区负载、光伏发电预测功率，单位：kW |

每张图同时输出到 `outputs/figures/`，提供 300 dpi PNG（3300 × 1590 像素）和可缩放 SVG 矢量图。小区负载为蓝色实线，光伏发电预测功率为橙色虚线，便于对比和黑白打印时识别。

时间轴沿用附件原始标签，从 `00:10` 到 `24:00`，相邻记录间隔 10 分钟；按题面附录 2，将 `0:00+1` 转为当天 `24:00`。坐标轴显示完整的 `00:00-24:00` 范围，但不额外补造 `00:00` 的数据点。折线直接连接原始点，不平滑、不降采样，也不将功率换算为电量。

绘图脚本检查表头、时间顺序、时间间隔、记录数量及数值有效性。此处绘图按数据标签定位；[问题一建模文档](docs/Question_1.md) 中的区间结束时刻假设用于后续调度计算，无需对绘图时间再平移。

建模符号参见 [symbols.md](docs/symbols.md)。
