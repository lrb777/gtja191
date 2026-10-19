"""Alpha191 专用函数"""
import numpy as np
from .ts_operators import DELAY, MAX, TSMAX, TSMIN, SUMIF


def DTM(OPEN, HIGH, LOW, CLOSE, n):
    """上涨开盘缺口指标。

    当 OPEN(t) > CLOSE(t-1) 时:
      S = MAX(OPEN(t), CLOSE(t-1))
      DTM = MAX(S - S(t-1), 0)
    否则返回0
    """
    S = np.maximum(OPEN, DELAY(CLOSE, 1))
    delta_S = S - DELAY(S, 1)
    cond = OPEN > DELAY(CLOSE, 1)
    return np.where(cond, np.maximum(delta_S, 0), 0.0)


def DBM(OPEN, HIGH, LOW, CLOSE, n):
    """下跌开盘缺口指标。

    当 OPEN(t) < CLOSE(t-1) 时:
      DBM = OPEN(t) - CLOSE(t-1)（负值，保持方向）
    否则返回0
    """
    cond = OPEN < DELAY(CLOSE, 1)
    return np.where(cond, OPEN - DELAY(CLOSE, 1), 0.0)


def SELF(prev_factor, date_idx, init_value=None):
    """SELF: 前一日因子值。

    初值需显式指定，不能为0（否则全序列为0）。
    """
    result = np.full_like(prev_factor, np.nan, dtype=np.float64)
    if init_value is not None:
        result[0] = init_value
    result[1:] = prev_factor[:-1]
    return result
