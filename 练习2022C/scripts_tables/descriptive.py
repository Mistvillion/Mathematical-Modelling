"""问题1第二小问：含量与log-ratio统计分开报告（p8-11）。"""
import numpy as np
import pandas as pd
from .constants import COMPONENTS
from .data import save_table


# 按玻璃类型和风化标签统计不同数据空间的成分分布并保存结果
def analyze(data, output):
    rows = []
    for basis in ['文物风化', '点位风化']:
        for (glass, weather), group in data.samples.groupby(['类型', basis], sort=False):
            for space in ['raw', 'closed', 'paper_clr']:
                x = pd.DataFrame(data.matrix(group, space), columns=COMPONENTS)
                for chemical in COMPONENTS:
                    col = x[chemical]
                    mean, std = col.mean(), col.std(ddof=1)
                    # CLR不是比率尺度，绝不能把std/负均值当可解释CV。
                    cv = std / mean if space != 'paper_clr' and mean > 0 else np.nan
                    rows.append({'分组口径': basis, '类型': glass, '风化': weather,
                                 '数据空间': space, '成分': chemical, 'n': len(col),
                                 '均值': mean, '标准差': std, '最小值': col.min(),
                                 'Q1': col.quantile(.25), '中位数': col.median(),
                                 'Q3': col.quantile(.75), '最大值': col.max(),
                                 '偏度': col.skew() if std > 0 else np.nan,
                                 '超额峰度': col.kurt() if std > 0 else np.nan, 'CV': cv})
    result = pd.DataFrame(rows)
    save_table(result, output, '12_四组描述统计')
    return result
