# coding=utf-8
"""Alpha191 因子实现 — 第4批 (Alpha151~Alpha191)"""
import numpy as np
from src.lib.ts_operators import *
from src.lib.cross_section import RANK, REGBETA, REGRESI, SEQUENCE


def alpha151(open, high, low, close, vol, amount, vwap, **kw):
    # [A] SMA(CLOSE-DELAY(CLOSE,20), 20, 1)
    return SMA(close - DELAY(close, 20), 20, 1)


def alpha152(open, high, low, close, vol, amount, vwap, **kw):
    # [A] SMA(MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),12) - MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),26), 9, 1)
    dc9 = DELAY(close, 9)
    ratio = close / dc9
    sma1 = SMA(DELAY(ratio, 1), 9, 1)
    delayed = DELAY(sma1, 1)
    return SMA(MEAN(delayed, 12) - MEAN(delayed, 26), 9, 1)


def alpha153(open, high, low, close, vol, amount, vwap, **kw):
    # [A] (MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/4
    return (MEAN(close, 3) + MEAN(close, 6) + MEAN(close, 12) + MEAN(close, 24)) / 4


def alpha154(open, high, low, close, vol, amount, vwap, **kw):
    # [B] MIN为滚动
    # (VWAP-TSMIN(VWAP,16)) < CORR(VWAP,MEAN(VOLUME,180),18)
    return ((vwap - TSMIN(vwap, 16)) < CORR(vwap, MEAN(vol, 180), 18)).astype(float)


def alpha155(open, high, low, close, vol, amount, vwap, **kw):
    # [A] SMA(VOLUME,13,2)-SMA(VOLUME,27,2)-SMA(SMA(VOLUME,13,2)-SMA(VOLUME,27,2),10,2)
    sma13 = SMA(vol, 13, 2)
    sma27 = SMA(vol, 27, 2)
    diff = sma13 - sma27
    return diff - SMA(diff, 10, 2)


def alpha156(open, high, low, close, vol, amount, vwap, **kw):
    # [A] MAX(RANK(DECAYLINEAR(DELTA(VWAP,5),3)), RANK(DECAYLINEAR((DELTA(OPEN*0.15+LOW*0.85,2)/(OPEN*0.15+LOW*0.85))*-1,3))) * -1
    inner1 = DELTA(vwap, 5)
    combo = open * 0.15 + low * 0.85
    inner2 = DELTA(combo, 2) / combo * -1
    return -1 * np.maximum(
        RANK(DECAYLINEAR(inner1, 3)),
        RANK(DECAYLINEAR(inner2, 3))
    )


def alpha157(open, high, low, close, vol, amount, vwap, ret=None, **kw):
    # [A] MIN(PROD(RANK(RANK(LOG(SUM(TSMIN(RANK(RANK(-1*RANK(DELTA((CLOSE-1),5)))),2),1)))),1),5) + TSRANK(DELAY(-1*RET,6),5)
    if ret is None:
        ret = close / DELAY(close, 1) - 1
    inner = -1 * RANK(DELTA(close - 1, 5))
    inner = RANK(RANK(inner))
    inner = TSMIN(inner, 2)
    inner = SUM(inner, 1)
    inner = LOG(inner)
    inner = RANK(RANK(inner))
    inner = PROD(inner, 1)
    part1 = TSMIN(inner, 5)
    part2 = TSRANK(DELAY(-1 * ret, 6), 5)
    return part1 + part2


def alpha158(open, high, low, close, vol, amount, vwap, **kw):
    # [D] 两SMA项抵消, 等于(HIGH-LOW)/CLOSE
    sma15 = SMA(close, 15, 2)
    return ((high - sma15) - (low - sma15)) / close


def alpha159(open, high, low, close, vol, amount, vwap, **kw):
    # [C] HGIH→HIGH, 第三权重候选6*12
    dc1 = DELAY(close, 1)
    hl = high  # HGIH→HIGH
    max_hl = np.maximum(hl, dc1)
    min_l = np.minimum(low, dc1)
    range6 = SUM(max_hl - min_l, 6)
    range12 = SUM(max_hl - min_l, 12)
    range24 = SUM(max_hl - min_l, 24)
    with np.errstate(invalid='ignore', divide='ignore'):
        val = ((close - SUM(min_l, 6)) / range6 * 12 * 24 +
               (close - SUM(min_l, 12)) / range12 * 6 * 24 +
               (close - SUM(min_l, 24)) / range24 * 6 * 24)
        return val * 100 / (6 * 12 + 6 * 24 + 12 * 24)


def alpha160(open, high, low, close, vol, amount, vwap, **kw):
    # [A] SMA((CLOSE<=DELAY(CLOSE,1)?STD(CLOSE,20):0), 20, 1)
    dc1 = DELAY(close, 1)
    std20 = STD(close, 20)
    cond = close <= dc1
    cond = np.where(np.isnan(cond), False, cond)
    val = np.where(cond, std20, 0.0)
    val = np.where(np.isnan(std20), 0.0, val)
    return SMA(val, 20, 1)


def alpha161(open, high, low, close, vol, amount, vwap, **kw):
    # [A] MEAN(MAX(MAX((HIGH-LOW),ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)), 12)
    dc1 = DELAY(close, 1)
    tr = np.maximum(np.maximum(high - low, ABS(dc1 - high)), ABS(dc1 - low))
    return MEAN(tr, 12)


def alpha162(open, high, low, close, vol, amount, vwap, **kw):
    # [B] MIN为滚动
    # (RSI - TSMIN(RSI,12)) / (TSMAX(RSI,12) - TSMIN(RSI,12))
    dc1 = DELAY(close, 1)
    up = np.maximum(close - dc1, 0)
    abs_val = ABS(close - dc1)
    with np.errstate(invalid='ignore', divide='ignore'):
        rsi = SMA(up, 12, 1) / SMA(abs_val, 12, 1) * 100
    max_rsi = TSMAX(rsi, 12)
    min_rsi = TSMIN(rsi, 12)
    with np.errstate(invalid='ignore', divide='ignore'):
        return (rsi - min_rsi) / (max_rsi - min_rsi)


def alpha163(open, high, low, close, vol, amount, vwap, ret=None, **kw):
    # [A] RANK((-1*RET*MEAN(VOLUME,20)*VWAP*(HIGH-CLOSE)))
    if ret is None:
        ret = close / DELAY(close, 1) - 1
    return RANK(-1 * ret * MEAN(vol, 20) * vwap * (high - close))


def alpha164(open, high, low, close, vol, amount, vwap, **kw):
    # [A] SMA(((CLOSE>DELAY(CLOSE,1)?1/(CLOSE-DELAY(CLOSE,1)):1) - MIN((CLOSE>DELAY(CLOSE,1)?1/(CLOSE-DELAY(CLOSE,1)):1),12))/(HIGH-LOW)*100, 13, 2)
    dc1 = DELAY(close, 1)
    with np.errstate(invalid='ignore', divide='ignore'):
        val = np.where(close > dc1, 1 / (close - dc1), 1)
    min_val = TSMIN(val, 12)
    with np.errstate(invalid='ignore', divide='ignore'):
        inner = (val - min_val) / (high - low) * 100
    return SMA(inner, 13, 2)


def alpha165(open, high, low, close, vol, amount, vwap, **kw):
    # [C] SUMAC缺窗口, 候选48
    dev = close - MEAN(close, 48)
    cumsum = SUMAC(dev, 48)
    with np.errstate(invalid='ignore', divide='ignore'):
        return (TSMAX(cumsum, 48) - TSMIN(cumsum, 48)) / STD(close, 48)


def alpha166(open, high, low, close, vol, amount, vwap, **kw):
    # [C] 表达式损坏, 候选: 偏度指标
    # -20*(20-1)^1.5 * SUM(r-mean,20) / ((20-1)*(20-2)*SUM(r^2,20)^1.5)
    # r = CLOSE/DELAY(CLOSE,1)-1
    dc1 = DELAY(close, 1)
    with np.errstate(invalid='ignore', divide='ignore'):
        r = close / dc1 - 1
    mean_r = MEAN(r, 20)
    sum_dev = SUM(r - mean_r, 20)
    sum_sq = SUM(r ** 2, 20)
    n = 20
    with np.errstate(invalid='ignore', divide='ignore'):
        num = -n * (n - 1) ** 1.5 * sum_dev
        den = (n - 1) * (n - 2) * sum_sq ** 1.5
        return np.where(den > 0, num / den, np.nan)


def alpha167(open, high, low, close, vol, amount, vwap, **kw):
    # [A] SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0), 12)
    dc1 = DELAY(close, 1)
    return SUM(np.where(close - dc1 > 0, close - dc1, 0), 12)


def alpha168(open, high, low, close, vol, amount, vwap, **kw):
    # [A] -1*VOLUME/MEAN(VOLUME,20)
    return -1 * vol / MEAN(vol, 20)


def alpha169(open, high, low, close, vol, amount, vwap, **kw):
    # [A] SMA(MEAN(DELAY(SMA(CLOSE-DELAY(CLOSE,1),9,1),1),12) - MEAN(DELAY(SMA(CLOSE-DELAY(CLOSE,1),9,1),1),26), 10, 1)
    dc1 = DELAY(close, 1)
    sma1 = SMA(close - dc1, 9, 1)
    delayed = DELAY(sma1, 1)
    return SMA(MEAN(delayed, 12) - MEAN(delayed, 26), 10, 1)


def alpha170(open, high, low, close, vol, amount, vwap, **kw):
    # [A] ((((RANK(1/CLOSE)*VOLUME)/MEAN(VOLUME,20))*((HIGH*RANK(HIGH-CLOSE))/(SUM(HIGH,5)/5))) - RANK(VWAP-DELAY(VWAP,5)))
    part1 = RANK(1 / close) * vol / MEAN(vol, 20)
    part2 = high * RANK(high - close) / (SUM(high, 5) / 5)
    return part1 * part2 - RANK(vwap - DELAY(vwap, 5))


def alpha171(open, high, low, close, vol, amount, vwap, **kw):
    # [A] -1*((LOW-CLOSE)*(OPEN^5))/((CLOSE-HIGH)*(CLOSE^5))
    with np.errstate(invalid='ignore', divide='ignore'):
        return -1 * (low - close) * (open ** 5) / ((close - high) * (close ** 5))


def alpha172(open, high, low, close, vol, amount, vwap, **kw):
    # [A] HD=HIGH-DELAY(HIGH,1), LD=DELAY(LOW,1)-LOW, TR=MAX(ABS(HD),ABS(LD),ABS(HIGH-DELAY(LOW,1)))
    dh = high - DELAY(high, 1)
    ld = DELAY(low, 1) - low
    dc1 = DELAY(close, 1)
    tr = np.maximum(np.maximum(ABS(dh), ABS(ld)), ABS(high - DELAY(low, 1)))
    with np.errstate(invalid='ignore', divide='ignore'):
        up_sum = SUM(np.where((ld > 0) & (ld > dh), ld, 0), 14) * 100 / SUM(tr, 14)
        dn_sum = SUM(np.where((dh > 0) & (dh > ld), dh, 0), 14) * 100 / SUM(tr, 14)
        val = ABS(up_sum - dn_sum) / (up_sum + dn_sum) * 100
    return MEAN(val, 6)


def alpha173(open, high, low, close, vol, amount, vwap, **kw):
    # [D] 3*SMA(CLOSE,13,2)-2*SMA(SMA(CLOSE,13,2),13,2)+SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2)
    sma1 = SMA(close, 13, 2)
    sma2 = SMA(sma1, 13, 2)
    log_sma = SMA(LOG(close), 13, 2)
    log_sma2 = SMA(log_sma, 13, 2)
    log_sma3 = SMA(log_sma2, 13, 2)
    return 3 * sma1 - 2 * sma2 + log_sma3


def alpha174(open, high, low, close, vol, amount, vwap, **kw):
    # [A] SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0), 20, 1)
    dc1 = DELAY(close, 1)
    std20 = STD(close, 20)
    cond = np.where(np.isnan(close > dc1), False, close > dc1)
    val = np.where(cond, np.where(np.isnan(std20), 0, std20), 0.0)
    return SMA(val, 20, 1)


def alpha175(open, high, low, close, vol, amount, vwap, **kw):
    # [A] MEAN(MAX(MAX((HIGH-LOW),ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)), 6)
    dc1 = DELAY(close, 1)
    tr = np.maximum(np.maximum(high - low, ABS(dc1 - high)), ABS(dc1 - low))
    return MEAN(tr, 6)


def alpha176(open, high, low, close, vol, amount, vwap, **kw):
    # [A] CORR(RANK((CLOSE-TSMIN(LOW,12))/(TSMAX(HIGH,12)-TSMIN(LOW,12))), RANK(VOLUME), 6)
    range12 = TSMAX(high, 12) - TSMIN(low, 12)
    with np.errstate(invalid='ignore', divide='ignore'):
        inner = np.where(range12 > 0, (close - TSMIN(low, 12)) / range12, 0)
    return CORR(RANK(inner), RANK(vol), 6)


def alpha177(open, high, low, close, vol, amount, vwap, **kw):
    # [A] ((20-HIGHDAY(HIGH,20))/20)*100
    return ((20 - HIGHDAY(high, 20)) / 20) * 100


def alpha178(open, high, low, close, vol, amount, vwap, **kw):
    # [A] (CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)*VOLUME
    dc1 = DELAY(close, 1)
    return (close - dc1) / dc1 * vol


def alpha179(open, high, low, close, vol, amount, vwap, **kw):
    # [A] RANK(CORR(VWAP,VOLUME,4)) * RANK(CORR(RANK(LOW),RANK(MEAN(VOLUME,50)),12))
    part1 = RANK(CORR(vwap, vol, 4))
    part2 = RANK(CORR(RANK(low), RANK(MEAN(vol, 50)), 12))
    return part1 * part2


def alpha180(open, high, low, close, vol, amount, vwap, **kw):
    # [A] (MEAN(VOLUME,20)<VOLUME) ? (-1*TSRANK(ABS(DELTA(CLOSE,7)),60)*SIGN(DELTA(CLOSE,7))) : (-1*VOLUME)
    return np.where(
        MEAN(vol, 20) < vol,
        -1 * TSRANK(ABS(DELTA(close, 7)), 60) * SIGN(DELTA(close, 7)),
        -1 * vol
    )


def alpha181(open, high, low, close, vol, amount, vwap, benchmark_close=None, **kw):
    # [C] SUM缺窗口, 指数结构需核实
    if benchmark_close is None:
        return np.full_like(close, np.nan)
    dc1 = DELAY(close, 1)
    stock_ret = close / dc1 - 1
    mean_sr = MEAN(stock_ret, 20)
    mean_bc = MEAN(benchmark_close, 20)
    dev = (stock_ret - mean_sr) - (benchmark_close - mean_bc)
    with np.errstate(invalid='ignore', divide='ignore'):
        return SUM(dev ** 2, 20) / SUM((benchmark_close - mean_bc) ** 3, 20)


def alpha182(open, high, low, close, vol, amount, vwap, benchmark_open=None, benchmark_close=None, **kw):
    # [B] 基准变量名反
    if benchmark_open is None or benchmark_close is None:
        return np.full_like(close, np.nan)
    cond = ((close > open) & (benchmark_close > benchmark_open)) | \
           ((close < open) & (benchmark_close < benchmark_open))
    return COUNT(cond, 20) / 20


def alpha183(open, high, low, close, vol, amount, vwap, **kw):
    # [C] 同Alpha165, 窗口24
    dev = close - MEAN(close, 24)
    cumsum = SUMAC(dev, 24)
    with np.errstate(invalid='ignore', divide='ignore'):
        return (TSMAX(cumsum, 24) - TSMIN(cumsum, 24)) / STD(close, 24)


def alpha184(open, high, low, close, vol, amount, vwap, **kw):
    # [A] RANK(CORR(DELAY(OPEN-CLOSE,1), CLOSE, 200)) + RANK(OPEN-CLOSE)
    return RANK(CORR(DELAY(open - close, 1), close, 200)) + RANK(open - close)


def alpha185(open, high, low, close, vol, amount, vwap, **kw):
    # [A] RANK(-1*((1-(OPEN/CLOSE))^2))
    with np.errstate(invalid='ignore', divide='ignore'):
        inner = -1 * (1 - open / close) ** 2
    return RANK(inner)


def alpha186(open, high, low, close, vol, amount, vwap, **kw):
    # [A] 同Alpha172 + DELAY(...,6)/2
    dh = high - DELAY(high, 1)
    ld = DELAY(low, 1) - low
    tr = np.maximum(np.maximum(ABS(dh), ABS(ld)), ABS(high - DELAY(low, 1)))
    with np.errstate(invalid='ignore', divide='ignore'):
        up_sum = SUM(np.where((ld > 0) & (ld > dh), ld, 0), 14) * 100 / SUM(tr, 14)
        dn_sum = SUM(np.where((dh > 0) & (dh > ld), dh, 0), 14) * 100 / SUM(tr, 14)
        val = ABS(up_sum - dn_sum) / (up_sum + dn_sum) * 100
    return (MEAN(val, 6) + DELAY(MEAN(val, 6), 6)) / 2


def alpha187(open, high, low, close, vol, amount, vwap, **kw):
    # [A] SUM((OPEN<=DELAY(OPEN,1)?0:MAX(HIGH-OPEN,OPEN-DELAY(OPEN,1))), 20)
    do1 = DELAY(open, 1)
    val = np.where(open <= do1, 0, np.maximum(high - open, open - do1))
    return SUM(val, 20)


def alpha188(open, high, low, close, vol, amount, vwap, **kw):
    # [A] ((HIGH-LOW-SMA(HIGH-LOW,11,2))/SMA(HIGH-LOW,11,2))*100
    hl = high - low
    sma = SMA(hl, 11, 2)
    with np.errstate(invalid='ignore', divide='ignore'):
        return (hl - sma) / sma * 100


def alpha189(open, high, low, close, vol, amount, vwap, **kw):
    # [A] MEAN(ABS(CLOSE-MEAN(CLOSE,6)), 6)
    return MEAN(ABS(close - MEAN(close, 6)), 6)


def alpha190(open, high, low, close, vol, amount, vwap, **kw):
    # [C] 多处DELAY缺期数, 补DELAY(CLOSE,1)
    # LOG((COUNT(r>g,20)-1)*(SUMIF((r-g)^2,20,r<g))/((COUNT(r<g,20))*(SUMIF((r-g)^2,20,r>g))))
    # r=CLOSE/DELAY(CLOSE,1)-1, g=(CLOSE/DELAY(CLOSE,19))^(1/20)-1
    dc1 = DELAY(close, 1)
    dc19 = DELAY(close, 19)
    with np.errstate(invalid='ignore', divide='ignore'):
        r = close / dc1 - 1
        g = np.where(dc19 > 0, dc19 ** (1.0/20) - 1, np.nan)
    diff = r - g
    cond_up = r > g
    cond_dn = r < g
    count_up = COUNT(cond_up, 20)
    count_dn = COUNT(cond_dn, 20)
    sumif_up = SUMIF(diff ** 2, cond_up, 20)
    sumif_dn = SUMIF(diff ** 2, cond_dn, 20)
    with np.errstate(invalid='ignore', divide='ignore'):
        num = (count_up - 1) * sumif_dn
        den = count_dn * sumif_up
        val = np.where(den > 0, num / den, np.nan)
        val = np.where(val > 0, LOG(val), np.nan)
    return val


def alpha191(open, high, low, close, vol, amount, vwap, **kw):
    # [D] CORR(MEAN(VOLUME,20),LOW,5) + (HIGH+LOW)/2 - CLOSE
    return CORR(MEAN(vol, 20), low, 5) + (high + low) / 2 - close
