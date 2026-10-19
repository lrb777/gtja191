"""
截面算子（每日跨股票操作）

输入: (T, N) 形状的 numpy 数组（行=交易日, 列=股票）
输出: 同形状数组
"""
import numpy as np


def RANK(S):
    """截面排名，归一化到[0,1]，从0开始，并列取平均。

    对每个交易日（行），对所有股票排名。
    """
    result = np.full_like(S, np.nan, dtype=np.float64)
    T, N = S.shape
    for t in range(T):
        row = S[t]
        valid = ~np.isnan(row)
        n_valid = valid.sum()
        if n_valid < 2:
            result[t, valid] = 0.5
            continue
        # 使用平均排名处理并列
        ranks = np.full(N, np.nan)
        ranks[valid] = row[valid].argsort().argsort().astype(float)
        # 处理并列：对相同值取平均排名
        vals = row[valid]
        sorted_idx = np.argsort(vals)
        sorted_vals = vals[sorted_idx]
        # 找到并列组
        i = 0
        avg_ranks = ranks.copy()
        valid_indices = np.where(valid)[0]
        while i < len(sorted_vals):
            j = i
            while j + 1 < len(sorted_vals) and sorted_vals[j + 1] == sorted_vals[i]:
                j += 1
            if j > i:
                avg_rank = (i + j) / 2.0
                for k in range(i, j + 1):
                    orig_idx = valid_indices[sorted_idx[k]]
                    avg_ranks[orig_idx] = avg_rank
            i = j + 1
        # 归一化到[0,1]
        n_valid = valid.sum()
        if n_valid > 1:
            result[t] = avg_ranks / (n_valid - 1)
        else:
            result[t, valid] = 0.5
    return result


def REGBETA(S, X, n):
    """OLS回归系数β（含截距），窗口n。

    S: 因变量 (T, N)
    X: 自变量 (T, N) 或 (T,)（单序列如SEQUENCE）
    n: 回归窗口

    返回: 每个时点最近的β值 (T, N)
    """
    result = np.full_like(S, np.nan, dtype=np.float64)
    T, N = S.shape

    if X.ndim == 1:
        X = np.tile(X[:, None], (1, N))

    for t in range(n - 1, T):
        seg_s = S[t - n + 1:t + 1]
        seg_x = X[t - n + 1:t + 1]
        for j in range(N):
            ys = seg_s[:, j]
            xs = seg_x[:, j]
            mask = ~(np.isnan(ys) | np.isnan(xs))
            cnt = mask.sum()
            if cnt < 3:
                continue
            xs_v = xs[mask]
            ys_v = ys[mask]
            x_mean = xs_v.mean()
            y_mean = ys_v.mean()
            ss_xx = np.sum((xs_v - x_mean) ** 2)
            if ss_xx == 0:
                result[t, j] = 0.0
            else:
                ss_xy = np.sum((xs_v - x_mean) * (ys_v - y_mean))
                result[t, j] = ss_xy / ss_xx
    return result


def REGRESI(S, X, n):
    """OLS残差（含截距），返回窗口最近一期的残差。

    返回: 每个时点最近一期的残差值 (T, N)
    """
    result = np.full_like(S, np.nan, dtype=np.float64)
    T, N = S.shape

    if X.ndim == 1:
        X = np.tile(X[:, None], (1, N))

    for t in range(n - 1, T):
        seg_s = S[t - n + 1:t + 1]
        seg_x = X[t - n + 1:t + 1]
        for j in range(N):
            ys = seg_s[:, j]
            xs = seg_x[:, j]
            mask = ~(np.isnan(ys) | np.isnan(xs))
            cnt = mask.sum()
            if cnt < 3:
                continue
            xs_v = xs[mask]
            ys_v = ys[mask]
            x_mean = xs_v.mean()
            y_mean = ys_v.mean()
            ss_xx = np.sum((xs_v - x_mean) ** 2)
            if ss_xx == 0:
                beta = 0.0
            else:
                beta = np.sum((xs_v - x_mean) * (ys_v - y_mean)) / ss_xx
            alpha = y_mean - beta * x_mean
            # 最近一期残差
            if mask[-1]:
                result[t, j] = ys[-1] - (alpha + beta * xs[-1])
            else:
                result[t, j] = np.nan
    return result


def SEQUENCE(n):
    """生成 1, 2, ..., n 的序列"""
    return np.arange(1, n + 1, dtype=np.float64)
