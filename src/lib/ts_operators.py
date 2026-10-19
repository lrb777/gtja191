"""
时序基础算子（纯numpy向量化实现）

用 sliding_window_view + numpy 向量化，避免 pandas rolling.apply 的开销。
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _rolling_view(S, n):
    """返回 (T-n+1, n, N) 的滑动窗口视图"""
    return sliding_window_view(S, n, axis=0)


# =====================================================================
# 一阶差分与滞后
# =====================================================================

def DELTA(S, d):
    result = np.full_like(S, np.nan, dtype=np.float64)
    result[d:] = S[d:] - S[:-d]
    return result


def DELAY(S, d):
    result = np.full_like(S, np.nan, dtype=np.float64)
    if d == 0:
        return S.copy()
    result[d:] = S[:-d]
    return result


# =====================================================================
# 滚动窗口聚合
# =====================================================================

def SUM(S, n):
    T, N = S.shape
    result = np.full_like(S, np.nan, dtype=np.float64)
    if T < n:
        return result
    # cumsum trick
    s_filled = np.nan_to_num(S, nan=0.0)
    csum = np.cumsum(s_filled, axis=0)
    result[n-1:] = csum[n-1:] - np.concatenate([np.zeros((1, N)), csum[:-n]], axis=0)
    # 窗口内有NaN → 结果设NaN
    nan_mask = np.isnan(S).astype(float)
    nan_count = np.cumsum(nan_mask, axis=0)
    nan_in_window = nan_count[n-1:] - np.concatenate([np.zeros((1, N)), nan_count[:-n]], axis=0)
    result[n-1:][nan_in_window > 0] = np.nan
    return result


def MEAN(S, n):
    T, N = S.shape
    result = np.full_like(S, np.nan, dtype=np.float64)
    if T < n:
        return result
    s = SUM(S, n)
    cnt = np.full_like(S, float(n))
    nan_count = np.cumsum(np.isnan(S).astype(float), axis=0)
    nan_in_window = nan_count[n-1:] - np.concatenate([np.zeros((1, N)), nan_count[:-n]], axis=0)
    cnt[n-1:] = n - nan_in_window
    with np.errstate(invalid='ignore', divide='ignore'):
        result = s / cnt
    result[cnt == 0] = np.nan
    return result


def STD(S, n, ddof=1):
    T, N = S.shape
    result = np.full_like(S, np.nan, dtype=np.float64)
    if T < n:
        return result
    m = MEAN(S, n)
    sq = S ** 2
    s_sum = SUM(S, n)
    sq_sum = SUM(sq, n)
    cnt = np.full_like(S, float(n))
    nan_count = np.cumsum(np.isnan(S).astype(float), axis=0)
    nan_in_window = nan_count[n-1:] - np.concatenate([np.zeros((1, N)), nan_count[:-n]], axis=0)
    cnt[n-1:] = n - nan_in_window
    with np.errstate(invalid='ignore', divide='ignore'):
        var = (sq_sum - s_sum * s_sum / cnt) / (cnt - ddof)
        var = np.maximum(var, 0)
        result = np.sqrt(var)
    result[cnt <= ddof] = np.nan
    return result


def COVIANCE(S1, S2, n, ddof=1):
    T, N = S1.shape
    result = np.full_like(S1, np.nan, dtype=np.float64)
    if T < n:
        return result
    m1 = MEAN(S1, n)
    m2 = MEAN(S2, n)
    cross_sum = SUM(S1 * S2, n)
    cnt = np.full_like(S1, float(n))
    nan_mask = np.isnan(S1) | np.isnan(S2)
    nan_count = np.cumsum(nan_mask.astype(float), axis=0)
    nan_in_window = nan_count[n-1:] - np.concatenate([np.zeros((1, N)), nan_count[:-n]], axis=0)
    cnt[n-1:] = n - nan_in_window
    with np.errstate(invalid='ignore', divide='ignore'):
        cov = (cross_sum - cnt * m1 * m2) / (cnt - ddof)
        result = cov
    result[cnt <= ddof] = np.nan
    return result


def CORR(S1, S2, n):
    T, N = S1.shape
    result = np.full_like(S1, np.nan, dtype=np.float64)
    if T < n:
        return result
    m1 = MEAN(S1, n)
    m2 = MEAN(S2, n)
    s1 = STD(S1, n, ddof=0)
    s2 = STD(S2, n, ddof=0)
    cov = COVIANCE(S1, S2, n, ddof=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        result = cov / (s1 * s2)
    # 零方差 → 0
    result = np.where(np.isnan(result), 0.0, result)
    result = np.where(np.isinf(result), 0.0, result)
    return result


# =====================================================================
# 滚动极值
# =====================================================================

def TSMAX(S, n):
    T, N = S.shape
    result = np.full_like(S, np.nan, dtype=np.float64)
    if T < n:
        return result
    windows = _rolling_view(S, n)  # (T-n+1, N, n) — sliding_window_view on axis=0
    result[n-1:] = np.nanmax(windows, axis=-1)
    return result


def TSMIN(S, n):
    T, N = S.shape
    result = np.full_like(S, np.nan, dtype=np.float64)
    if T < n:
        return result
    windows = _rolling_view(S, n)
    result[n-1:] = np.nanmin(windows, axis=-1)
    return result


def MAX(S, n):
    return TSMAX(S, n)


def MIN(S, n):
    return TSMIN(S, n)


# =====================================================================
# 时序排名
# =====================================================================

def TSRANK(S, n):
    T, N = S.shape
    result = np.full_like(S, np.nan, dtype=np.float64)
    if T < n:
        return result
    windows = _rolling_view(S, n)  # (T-n+1, N, n)
    today = S[n-1:]  # (T-n+1, N)
    count_le = np.sum(windows <= today[:, :, None], axis=-1)  # (T-n+1, N)
    valid_count = np.sum(~np.isnan(windows), axis=-1)
    with np.errstate(invalid='ignore', divide='ignore'):
        result[n-1:] = count_le / valid_count
    result[n-1:][valid_count == 0] = np.nan
    return result


# =====================================================================
# 加权移动平均
# =====================================================================

def WMA(S, n):
    T, N = S.shape
    result = np.full_like(S, np.nan, dtype=np.float64)
    if T < n:
        return result
    weights = np.arange(1, n + 1, dtype=np.float64)[::-1]  # n, n-1, ..., 1
    windows = _rolling_view(S, n)  # (T-n+1, N, n)
    wgt = weights[None, None, :]  # (1, 1, n)
    wsum = np.sum(windows * wgt, axis=-1)
    result[n-1:] = wsum / weights.sum()
    return result


def DECAYLINEAR(S, n):
    return WMA(S, n)


# =====================================================================
# SMA 递推（无法向量化，但单列很快）
# =====================================================================

def SMA(S, n, m):
    T, N = S.shape
    result = np.full_like(S, np.nan, dtype=np.float64)
    ratio = m / n
    for j in range(N):
        col = S[:, j]
        first_valid = None
        for t in range(T):
            if not np.isnan(col[t]):
                if first_valid is None:
                    result[t, j] = col[t]
                    first_valid = t
                else:
                    result[t, j] = ratio * col[t] + (1 - ratio) * result[t - 1, j]
    return result


# =====================================================================
# 其他
# =====================================================================

def COUNT(cond, n):
    return SUM(cond.astype(float), n)


def PROD(S, n):
    T, N = S.shape
    result = np.full_like(S, np.nan, dtype=np.float64)
    if T < n:
        return result
    windows = _rolling_view(np.nan_to_num(S, nan=1.0), n)
    result[n-1:] = np.prod(windows, axis=-1)
    return result


def HIGHDAY(S, n):
    T, N = S.shape
    result = np.full_like(S, np.nan, dtype=np.float64)
    if T < n:
        return result
    windows = _rolling_view(S, n)  # (T-n+1, N, n)
    for t in range(windows.shape[0]):
        w = windows[t]  # (N, n)
        for j in range(N):
            col = w[j]
            valid = col[~np.isnan(col)]
            if len(valid) == 0:
                continue
            max_idx = np.nanargmax(col)
            result[t + n - 1, j] = (n - 1) - max_idx
    return result


def LOWDAY(S, n):
    T, N = S.shape
    result = np.full_like(S, np.nan, dtype=np.float64)
    if T < n:
        return result
    windows = _rolling_view(S, n)  # (T-n+1, N, n)
    for t in range(windows.shape[0]):
        w = windows[t]
        for j in range(N):
            col = w[j]
            valid = col[~np.isnan(col)]
            if len(valid) == 0:
                continue
            min_idx = np.nanargmin(col)
            result[t + n - 1, j] = (n - 1) - min_idx
    return result


def SUMIF(S, cond, n):
    masked = np.where(cond, S, 0.0)
    return SUM(masked, n)


def SUMAC(S, n):
    return SUM(S, n)


def FILTER(S, n):
    result = S.copy().astype(bool)
    T = S.shape[0]
    for t in range(T):
        if S[t].any():
            for k in range(1, n):
                if t + k < T:
                    result[t + k] = False
    return result


# =====================================================================
# 逐元素运算
# =====================================================================

def ABS(S):
    return np.abs(S)


def LOG(S):
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(S > 0, np.log(S), np.nan)


def SIGN(S):
    return np.sign(S)


def POWER(S, p):
    with np.errstate(invalid='ignore'):
        return np.power(S, p)


def SCALE(S):
    mean = np.nanmean(S, axis=1, keepdims=True)
    std = np.nanstd(S, axis=1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(std > 0, (S - mean) / std, 0.0)


def WHERE(cond, x, y):
    return np.where(cond, x, y)
