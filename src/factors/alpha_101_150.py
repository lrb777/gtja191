# coding=utf-8
"""Alpha191 因子实现 — 第3批 (Alpha101~Alpha150)"""
import numpy as np
from src.lib.ts_operators import *
from src.lib.cross_section import RANK, REGBETA, REGRESI, SEQUENCE


def alpha101(open, high, low, close, vol, amount, vwap, **kw):
    # [A] (RANK(CORR(CLOSE, SUM(MEAN(VOLUME,30),37), 15)) < RANK(CORR(RANK(HIGH*0.1+VWAP*0.9), RANK(VOLUME), 11))) * -1
    part1 = RANK(CORR(close, SUM(MEAN(vol, 30), 37), 15))
    part2 = RANK(CORR(RANK(high * 0.1 + vwap * 0.9), RANK(vol), 11))
    return ((part1 < part2).astype(float)) * -1


def alpha102(open, high, low, close, vol, amount, vwap, **kw):
    # [A] SMA(MAX(VOLUME-DELAY(VOLUME,1),0),6,1)/SMA(ABS(VOLUME-DELAY(VOLUME,1)),6,1)*100
    dv1 = DELAY(vol, 1)
    up = np.maximum(vol - dv1, 0)
    abs_val = ABS(vol - dv1)
    with np.errstate(invalid='ignore', divide='ignore'):
        return SMA(up, 6, 1) / SMA(abs_val, 6, 1) * 100


def alpha103(open, high, low, close, vol, amount, vwap, **kw):
    # [B] LOWDAY按函数名语义（最小值距今）
    return ((20 - LOWDAY(low, 20)) / 20) * 100


def alpha104(open, high, low, close, vol, amount, vwap, **kw):
    # [A] -1*(DELTA(CORR(HIGH,VOLUME,5),5) * RANK(STD(CLOSE,20)))
    return -1 * (DELTA(CORR(high, vol, 5), 5) * RANK(STD(close, 20)))


def alpha105(open, high, low, close, vol, amount, vwap, **kw):
    # [A] -1*CORR(RANK(OPEN), RANK(VOLUME), 10)
    return -1 * CORR(RANK(open), RANK(vol), 10)


def alpha106(open, high, low, close, vol, amount, vwap, **kw):
    # [A] CLOSE-DELAY(CLOSE,20)
    return close - DELAY(close, 20)


def alpha107(open, high, low, close, vol, amount, vwap, **kw):
    # [A] -1*RANK(OPEN-DELAY(HIGH,1)) * RANK(OPEN-DELAY(CLOSE,1)) * RANK(OPEN-DELAY(LOW,1))
    return -1 * RANK(open - DELAY(high, 1)) * RANK(open - DELAY(close, 1)) * RANK(open - DELAY(low, 1))


def alpha108(open, high, low, close, vol, amount, vwap, **kw):
    # [B] MIN为滚动
    # (RANK(HIGH-MIN(HIGH,2))^RANK(CORR(VWAP,MEAN(VOLUME,120),6))) * -1
    return -1 * RANK(high - TSMIN(high, 2)) ** RANK(CORR(vwap, MEAN(vol, 120), 6))


def alpha109(open, high, low, close, vol, amount, vwap, **kw):
    # [A] SMA(HIGH-LOW,10,2)/SMA(SMA(HIGH-LOW,10,2),10,2)
    hl = high - low
    sma1 = SMA(hl, 10, 2)
    return sma1 / SMA(sma1, 10, 2)


def alpha110(open, high, low, close, vol, amount, vwap, **kw):
    # [A] SUM(MAX(0,HIGH-DELAY(CLOSE,1)),20)/SUM(MAX(0,DELAY(CLOSE,1)-LOW),20)*100
    dc1 = DELAY(close, 1)
    return SUM(np.maximum(0, high - dc1), 20) / SUM(np.maximum(0, dc1 - low), 20) * 100


def alpha111(open, high, low, close, vol, amount, vwap, **kw):
    # [C] VOL→VOLUME
    # SMA(VOLUME*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),11,2) - SMA(VOLUME*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),4,2)
    with np.errstate(invalid='ignore', divide='ignore'):
        inner = vol * ((close - low) - (high - close)) / (high - low)
    return SMA(inner, 11, 2) - SMA(inner, 4, 2)


def alpha112(open, high, low, close, vol, amount, vwap, **kw):
    # [A] (SUM(涨)-SUM(跌))/(SUM(涨)+SUM(跌))*100
    dc1 = DELAY(close, 1)
    diff = close - dc1
    up = SUM(np.where(diff > 0, diff, 0), 12)
    dn = SUM(np.where(diff < 0, ABS(diff), 0), 12)
    with np.errstate(invalid='ignore', divide='ignore'):
        return (up - dn) / (up + dn) * 100


def alpha113(open, high, low, close, vol, amount, vwap, **kw):
    # [A] -1*(RANK(SUM(DELAY(CLOSE,5),20)/20) * CORR(CLOSE,VOLUME,2) * RANK(CORR(SUM(CLOSE,5),SUM(CLOSE,20),2)))
    part1 = RANK(SUM(DELAY(close, 5), 20) / 20)
    part2 = CORR(close, vol, 2)
    part3 = RANK(CORR(SUM(close, 5), SUM(close, 20), 2))
    return -1 * part1 * part2 * part3


def alpha114(open, high, low, close, vol, amount, vwap, **kw):
    # [A] RANK(DELAY((HIGH-LOW)/(SUM(CLOSE,5)/5),2)) * RANK(RANK(VOLUME)) / (((HIGH-LOW)/(SUM(CLOSE,5)/5))/(VWAP-CLOSE))
    hl_over_s5 = (high - low) / (SUM(close, 5) / 5)
    with np.errstate(invalid='ignore', divide='ignore'):
        return RANK(DELAY(hl_over_s5, 2)) * RANK(RANK(vol)) / (hl_over_s5 / (vwap - close))


def alpha115(open, high, low, close, vol, amount, vwap, **kw):
    # [A] RANK(CORR(HIGH*0.9+CLOSE*0.1, MEAN(VOLUME,30), 10))^RANK(CORR(TSRANK((HIGH+LOW)/2,4), TSRANK(VOLUME,10), 7))
    corr1 = CORR(high * 0.9 + close * 0.1, MEAN(vol, 30), 10)
    corr2 = CORR(TSRANK((high + low) / 2, 4), TSRANK(vol, 10), 7)
    return RANK(corr1) ** RANK(corr2)


def alpha116(open, high, low, close, vol, amount, vwap, **kw):
    # [C] SEQUENCE→SEQUENCE(20)
    T = close.shape[0]
    seq = np.tile(SEQUENCE(20), (T // 20 + 1,))[:T]
    return REGBETA(close, seq, 20)


def alpha117(open, high, low, close, vol, amount, vwap, ret=None, **kw):
    # [A] TSRANK(VOLUME,32) * (1-TSRANK((CLOSE+HIGH-LOW),16)) * (1-TSRANK(RET,32))
    if ret is None:
        ret = close / DELAY(close, 1) - 1
    return TSRANK(vol, 32) * (1 - TSRANK(close + high - low, 16)) * (1 - TSRANK(ret, 32))


def alpha118(open, high, low, close, vol, amount, vwap, **kw):
    # [A] SUM(HIGH-OPEN,20)/SUM(OPEN-LOW,20)*100
    return SUM(high - open, 20) / SUM(open - low, 20) * 100


def alpha119(open, high, low, close, vol, amount, vwap, **kw):
    # [B] MIN为滚动
    # RANK(DECAYLINEAR(CORR(VWAP,SUM(MEAN(VOLUME,5),26),5),7)) - RANK(DECAYLINEAR(TSRANK(MIN(CORR(RANK(OPEN),RANK(MEAN(VOLUME,15)),21),9),7),8))
    corr1 = CORR(vwap, SUM(MEAN(vol, 5), 26), 5)
    corr2 = CORR(RANK(open), RANK(MEAN(vol, 15)), 21)
    part1 = RANK(DECAYLINEAR(corr1, 7))
    part2 = RANK(DECAYLINEAR(TSRANK(TSMIN(corr2, 9), 7), 8))
    return part1 - part2


def alpha120(open, high, low, close, vol, amount, vwap, **kw):
    # [A] RANK(VWAP-CLOSE)/RANK(VWAP+CLOSE)
    return RANK(vwap - close) / RANK(vwap + close)


def alpha121(open, high, low, close, vol, amount, vwap, **kw):
    # [B] MIN为滚动
    # (RANK(VWAP-TSMIN(VWAP,12))^TSRANK(CORR(TSRANK(VWAP,20),TSRANK(MEAN(VOLUME,60),2),18),3)) * -1
    corr_val = CORR(TSRANK(vwap, 20), TSRANK(MEAN(vol, 60), 2), 18)
    return -1 * RANK(vwap - TSMIN(vwap, 12)) ** TSRANK(corr_val, 3)


def alpha122(open, high, low, close, vol, amount, vwap, **kw):
    # [A] (SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2)-DELAY(...,1))/DELAY(...,1)
    sma1 = SMA(LOG(close), 13, 2)
    sma2 = SMA(sma1, 13, 2)
    sma3 = SMA(sma2, 13, 2)
    delayed = DELAY(sma3, 1)
    with np.errstate(invalid='ignore', divide='ignore'):
        return (sma3 - delayed) / delayed


def alpha123(open, high, low, close, vol, amount, vwap, **kw):
    # [A] (RANK(CORR(SUM((HIGH+LOW)/2,20),SUM(MEAN(VOLUME,60),20),9)) < RANK(CORR(LOW,VOLUME,6))) * -1
    part1 = RANK(CORR(SUM((high + low) / 2, 20), SUM(MEAN(vol, 60), 20), 9))
    part2 = RANK(CORR(low, vol, 6))
    return ((part1 < part2).astype(float)) * -1


def alpha124(open, high, low, close, vol, amount, vwap, **kw):
    # [A] (CLOSE-VWAP)/DECAYLINEAR(RANK(TSMAX(CLOSE,30)),2)
    with np.errstate(invalid='ignore', divide='ignore'):
        return (close - vwap) / DECAYLINEAR(RANK(TSMAX(close, 30)), 2)


def alpha125(open, high, low, close, vol, amount, vwap, **kw):
    # [A] RANK(DECAYLINEAR(CORR(VWAP,MEAN(VOLUME,80),17),20)) / RANK(DECAYLINEAR(DELTA(CLOSE*0.5+VWAP*0.5,3),16))
    corr_val = CORR(vwap, MEAN(vol, 80), 17)
    return RANK(DECAYLINEAR(corr_val, 20)) / RANK(DECAYLINEAR(DELTA(close * 0.5 + vwap * 0.5, 3), 16))


def alpha126(open, high, low, close, vol, amount, vwap, **kw):
    # [A] (CLOSE+HIGH+LOW)/3
    return (close + high + low) / 3


def alpha127(open, high, low, close, vol, amount, vwap, **kw):
    # [C] MEAN缺窗口+MAX语义, 候选n=12, MAX→TSMAX
    # MEAN((100*(CLOSE-TSMAX(CLOSE,12))/TSMAX(CLOSE,12))^2)^(1/2)
    tsm = TSMAX(close, 12)
    with np.errstate(invalid='ignore', divide='ignore'):
        val = (100 * (close - tsm) / tsm) ** 2
    return np.sqrt(MEAN(val, 12))


def alpha128(open, high, low, close, vol, amount, vwap, **kw):
    # [A] 100-(100/(1+SUM(涨,14)/SUM(跌,14)))
    hlc3 = (high + low + close) / 3
    dhlc3 = DELAY(hlc3, 1)
    up = SUM(np.where(hlc3 > dhlc3, hlc3 * vol, 0), 14)
    dn = SUM(np.where(hlc3 < dhlc3, hlc3 * vol, 0), 14)
    with np.errstate(invalid='ignore', divide='ignore'):
        return 100 - 100 / (1 + up / dn)


def alpha129(open, high, low, close, vol, amount, vwap, **kw):
    # [A] SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0), 12)
    dc1 = DELAY(close, 1)
    return SUM(np.where(close - dc1 < 0, ABS(close - dc1), 0), 12)


def alpha130(open, high, low, close, vol, amount, vwap, **kw):
    # [A] RANK(DECAYLINEAR(CORR((HIGH+LOW)/2,MEAN(VOLUME,40),9),10)) / RANK(DECAYLINEAR(CORR(RANK(VWAP),RANK(VOLUME),7),3))
    corr1 = CORR((high + low) / 2, MEAN(vol, 40), 9)
    corr2 = CORR(RANK(vwap), RANK(vol), 7)
    return RANK(DECAYLINEAR(corr1, 10)) / RANK(DECAYLINEAR(corr2, 3))


def alpha131(open, high, low, close, vol, amount, vwap, **kw):
    # [C] DELAT→DELTA
    # RANK(DELTA(VWAP,1))^TSRANK(CORR(CLOSE,MEAN(VOLUME,50),18),18)
    corr_val = CORR(close, MEAN(vol, 50), 18)
    return RANK(DELTA(vwap, 1)) ** TSRANK(corr_val, 18)


def alpha132(open, high, low, close, vol, amount, vwap, **kw):
    # [A] MEAN(AMOUNT,20)
    return MEAN(amount, 20)


def alpha133(open, high, low, close, vol, amount, vwap, **kw):
    # [B] LOWDAY按函数名语义
    return ((20 - HIGHDAY(high, 20)) / 20) * 100 - ((20 - LOWDAY(low, 20)) / 20) * 100


def alpha134(open, high, low, close, vol, amount, vwap, **kw):
    # [A] (CLOSE-DELAY(CLOSE,12))/DELAY(CLOSE,12)*VOLUME
    dc12 = DELAY(close, 12)
    return (close - dc12) / dc12 * vol


def alpha135(open, high, low, close, vol, amount, vwap, **kw):
    # [A] SMA(DELAY(CLOSE/DELAY(CLOSE,20),1),20,1)
    dc20 = DELAY(close, 20)
    return SMA(DELAY(close / dc20, 1), 20, 1)


def alpha136(open, high, low, close, vol, amount, vwap, ret=None, **kw):
    # [A] -1*RANK(DELTA(RET,3)) * CORR(OPEN,VOLUME,10)
    if ret is None:
        ret = close / DELAY(close, 1) - 1
    return -1 * RANK(DELTA(ret, 3)) * CORR(open, vol, 10)


def alpha137(open, high, low, close, vol, amount, vwap, **kw):
    # [A] 复杂ATR变体, 同Alpha55但不做SUM
    dc1 = DELAY(close, 1)
    do1 = DELAY(open, 1)
    dl1 = DELAY(low, 1)
    tr1 = ABS(high - dc1)
    tr2 = ABS(low - dc1)
    tr3 = ABS(high - dl1)
    cond1 = (tr1 > tr2) & (tr1 > tr3)
    val1 = tr1 + tr2 / 2 + ABS(dc1 - do1) / 4
    cond2 = (tr2 > tr3) & (tr2 > tr1)
    val2 = tr2 + tr1 / 2 + ABS(dc1 - do1) / 4
    val3 = tr3 + ABS(dc1 - do1) / 4
    true_range = np.where(cond1, val1, np.where(cond2, val2, val3))
    max_tr = np.maximum(tr1, tr2)
    with np.errstate(invalid='ignore', divide='ignore'):
        return 16 * (close - dc1 + (close - open) / 2 + dc1 - do1) / (true_range * max_tr)


def alpha138(open, high, low, close, vol, amount, vwap, **kw):
    # [A] (RANK(DECAYLINEAR(DELTA(LOW*0.7+VWAP*0.3,3),20)) - TSRANK(DECAYLINEAR(TSRANK(CORR(TSRANK(LOW,8),TSRANK(MEAN(VOLUME,60),17),5),19),16),7)) * -1
    part1 = RANK(DECAYLINEAR(DELTA(low * 0.7 + vwap * 0.3, 3), 20))
    corr_val = CORR(TSRANK(low, 8), TSRANK(MEAN(vol, 60), 17), 5)
    part2 = TSRANK(DECAYLINEAR(TSRANK(corr_val, 19), 16), 7)
    return -1 * (part1 - part2)


def alpha139(open, high, low, close, vol, amount, vwap, **kw):
    # [A] -1*CORR(OPEN, VOLUME, 10)
    return -1 * CORR(open, vol, 10)


def alpha140(open, high, low, close, vol, amount, vwap, **kw):
    # [A] MIN(RANK(DECAYLINEAR(RANK(OPEN)+RANK(LOW)-RANK(HIGH)-RANK(CLOSE),8)), TSRANK(DECAYLINEAR(CORR(TSRANK(CLOSE,8),TSRANK(MEAN(VOLUME,60),20),8),7),3))
    inner1 = RANK(open) + RANK(low) - RANK(high) - RANK(close)
    corr_val = CORR(TSRANK(close, 8), TSRANK(MEAN(vol, 60), 20), 8)
    return np.minimum(
        RANK(DECAYLINEAR(inner1, 8)),
        TSRANK(DECAYLINEAR(corr_val, 7), 3)
    )


def alpha141(open, high, low, close, vol, amount, vwap, **kw):
    # [A] RANK(CORR(RANK(HIGH),RANK(MEAN(VOLUME,15)),9)) * -1
    return -1 * RANK(CORR(RANK(high), RANK(MEAN(vol, 15)), 9))


def alpha142(open, high, low, close, vol, amount, vwap, **kw):
    # [A] -1*RANK(TSRANK(CLOSE,10)) * RANK(DELTA(DELTA(CLOSE,1),1)) * RANK(TSRANK(VOLUME/MEAN(VOLUME,20),5))
    part1 = RANK(TSRANK(close, 10))
    part2 = RANK(DELTA(DELTA(close, 1), 1))
    part3 = RANK(TSRANK(vol / MEAN(vol, 20), 5))
    return -1 * part1 * part2 * part3


def alpha143(open, high, low, close, vol, amount, vwap, **kw):
    # [C] SELF无初值
    # CLOSE>DELAY(CLOSE,1)?(CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)*SELF:SELF
    # 初值=1.0（累计收益率起点）
    dc1 = DELAY(close, 1)
    ret = np.where(np.isnan(dc1) | (dc1 == 0), 0, (close - dc1) / dc1)
    # SELF = 前一日因子值，初值=1
    T, N = close.shape
    result = np.full_like(close, np.nan, dtype=np.float64)
    result[0] = 1.0
    for t in range(1, T):
        result[t] = np.where(close[t] > dc1[t], ret[t] * result[t-1], result[t-1])
    return result


def alpha144(open, high, low, close, vol, amount, vwap, **kw):
    # [A] SUMIF(ABS(CLOSE/DELAY(CLOSE,1)-1)/AMOUNT, 20, CLOSE<DELAY(CLOSE,1)) / COUNT(CLOSE<DELAY(CLOSE,1), 20)
    dc1 = DELAY(close, 1)
    with np.errstate(invalid='ignore', divide='ignore'):
        val = ABS(close / dc1 - 1) / amount
    cond = close < dc1
    s = SUMIF(val, cond, 20)
    c = COUNT(cond, 20)
    with np.errstate(invalid='ignore', divide='ignore'):
        return s / c


def alpha145(open, high, low, close, vol, amount, vwap, **kw):
    # [A] (MEAN(VOLUME,9)-MEAN(VOLUME,26))/MEAN(VOLUME,12)*100
    return (MEAN(vol, 9) - MEAN(vol, 26)) / MEAN(vol, 12) * 100


def alpha146(open, high, low, close, vol, amount, vwap, **kw):
    # [C] SMA缺m, 候选m=2
    dc1 = DELAY(close, 1)
    r = (close - dc1) / dc1
    sma_r = SMA(r, 61, 2)
    diff = r - sma_r
    inner = MEAN(diff, 20) * diff
    denom = SMA(diff ** 2, 60, 2)
    with np.errstate(invalid='ignore', divide='ignore'):
        return inner / denom


def alpha147(open, high, low, close, vol, amount, vwap, **kw):
    # [C] REGBETA缺窗口, SEQUENCE(12)→n=12
    T = close.shape[0]
    seq = np.tile(SEQUENCE(12), (T // 12 + 1,))[:T]
    return REGBETA(MEAN(close, 12), seq, 12)


def alpha148(open, high, low, close, vol, amount, vwap, **kw):
    # [A] (RANK(CORR(OPEN,SUM(MEAN(VOLUME,60),9),6)) < RANK(OPEN-TSMIN(OPEN,14))) * -1
    part1 = RANK(CORR(open, SUM(MEAN(vol, 60), 9), 6))
    part2 = RANK(open - TSMIN(open, 14))
    return ((part1 < part2).astype(float)) * -1


def alpha149(open, high, low, close, vol, amount, vwap, benchmark_close=None, **kw):
    # [B] 基准指数变量名反
    if benchmark_close is None:
        return np.full_like(close, np.nan)
    bc = benchmark_close
    dbc = DELAY(bc, 1)
    stock_ret = close / DELAY(close, 1) - 1
    bench_ret = bc / dbc - 1
    cond = bc < dbc
    # FILTER+REGBETA: 筛出下跌日后回归
    filtered_stock = np.where(cond, stock_ret, np.nan)
    filtered_bench = np.where(cond, bench_ret, np.nan)
    return REGBETA(filtered_stock, filtered_bench, 252)


def alpha150(open, high, low, close, vol, amount, vwap, **kw):
    # [A] (CLOSE+HIGH+LOW)/3*VOLUME
    return (close + high + low) / 3 * vol
