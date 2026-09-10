"""成分闭合、论文的非零子组成 log-ratio 以及带独立零掩码的逆变换。"""
import numpy as np


# 将输入转换为浮点矩阵并检查二维结构与有限值
def _matrix(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('需要有限值二维矩阵。')
    return x


# 将每个样本的成分含量按比例归一化到指定总量
def closure(x, total=100.0):
    x = _matrix(x)
    if (x < 0).any() or (x.sum(axis=1) <= 0).any() or total <= 0:
        raise ValueError('成分必须非负且每行至少有一个正数，总量须为正数。')
    return x / x.sum(axis=1, keepdims=True) * total


# 对非零成分计算中心化对数比并保留原始非零掩码
def paper_clr(x):
    """对应附录D：仅对每行非零成分取几何均值，零位填0；不是标准全维CLR。"""
    x = _matrix(x)
    closure(x)  # 验证非负与非空组成
    present = x > 0
    log_x = np.log(np.where(present, x, 1.0))
    mean_log = log_x.sum(axis=1, keepdims=True) / present.sum(axis=1, keepdims=True)
    return np.where(present, log_x - mean_log, 0.0), present


# 将对数比坐标还原为指定总量的成分含量并保留零位
def inverse_clr(z, present=None, total=100.0):
    """稳定softmax；存在性由原含量决定，不能用 z==0 推断未检出。"""
    z = _matrix(z)
    if present is None:
        present = np.ones_like(z, dtype=bool)
    present = np.asarray(present, dtype=bool)
    if present.shape != z.shape or not present.any(axis=1).all():
        raise ValueError('零掩码形状错误或存在全零组成。')
    masked = np.where(present, z, -np.inf)
    exp_z = np.exp(masked - masked.max(axis=1, keepdims=True))
    return closure(exp_z, total)


# 对零含量作乘法替换后计算标准中心化对数比
def replaced_clr(x, delta=1e-4):
    """补充对照：对闭合到1的组成作乘法零替换，再取标准CLR。

    delta 是替换后每个零的比例（默认0.01个百分点）；未检出不等于结构零，
    此选择仍需检出限信息和敏感性研究，不能当作唯一正确处理。
    """
    p = closure(x, 1.0)
    zero = p == 0
    n_zero = zero.sum(axis=1, keepdims=True)
    if delta <= 0 or (delta * n_zero >= 1).any():
        raise ValueError('delta必须为正且每行替换总量小于1。')
    p = np.where(zero, delta, p * (1 - n_zero * delta))
    logs = np.log(p)
    return logs - logs.mean(axis=1, keepdims=True)
