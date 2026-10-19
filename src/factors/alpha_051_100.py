# coding=utf-8
"""Alpha191 因子实现 — 第2批 (Alpha51~Alpha100)"""
import numpy as np
from src.lib.ts_operators import *
from src.lib.cross_section import RANK, REGBETA, REGRESI, SEQUENCE


def alpha51(open, high, low, close, vol, amount, vwap, **kw):
    # [D] SUM(dn,12)/(SUM(dn,12)+SUM(up,12))
    dh = DELAY(high, 1)
    dl = DELAY(low, 1)
    hl = high + low
    dhl = dh + dl
    dn_cond = hl <= dhl
    up_cond = hl >= dhl
    max_val = np.maximum(ABS(high - dh), ABS(low - dl))
    dn_sum = SUM(np.where(dn_cond, 0, max_val), 12)
    up_sum = SUM(np.where(up_cond, 0, max_val), 12)
    with np.errstate(invalid='ignore', divide='ignore'):
        return dn_sum / (dn_sum + up_sum)


def alpha52(open, high, low, close, vol, amount, vwap, **kw):
    # [C] L→LOW
    hlc3 = (high + low + close) / 3
    dhlc3 = DELAY(hlc3, 1)
    return SUM(np.maximum(0, high - dhlc3), 26) / SUM(np.maximum(0, dhlc3 - low), 26) * 100


def alpha53(open, high, low, close, vol, amount, vwap, **kw):
    # [A] COUNT(CLOSE>DELAY(CLOSE,1),12)/12*100
    return COUNT(close > DELAY(close, 1), 12) / 12 * 100


def alpha54(open, high, low, close, vol, amount, vwap, **kw):
    # [C] STD缺窗口, 候选n=20
    # -1*RANK(STD(ABS(CLOSE-OPEN),20) + (CLOSE-OPEN) + CORR(CLOSE,OPEN,10))
    return -1 * RANK(STD(ABS(close - open), 20) + (close - open) + CORR(close, open, 10))


def alpha55(open, high, low, close, vol, amount, vwap, **kw):
    # [A] 复杂条件表达式（ATR变体）
    dc1 = DELAY(close, 1)
    do1 = DELAY(open, 1)
    dl1 = DELAY(low, 1)
    dh1 = DELAY(high, 1)
    tr1 = ABS(high - dc1)
    tr2 = ABS(low - dc1)
    tr3 = ABS(high - dl1)
    # 条件1: tr1>tr2 & tr1>tr3
    cond1 = (tr1 > tr2) & (tr1 > tr3)
    val1 = tr1 + tr2 / 2 + ABS(dc1 - do1) / 4
    # 条件2: tr2>tr3 & tr2>tr1
    cond2 = (tr2 > tr3) & (tr2 > tr1)
    val2 = tr2 + tr1 / 2 + ABS(dc1 - do1) / 4
    # 默认
    val3 = tr3 + ABS(dc1 - do1) / 4
    true_range = np.where(cond1, val1, np.where(cond2, val2, val3))
    max_tr = np.maximum(tr1, tr2)
    with np.errstate(invalid='ignore', divide='ignore'):
        inner = 16 * (close - dc1 + (close - open) / 2 + dc1 - do1) / (true_range * max_tr)
    return SUM(inner, 20)


def alpha56(open, high, low, close, vol, amount, vwap, **kw):
    # [A] RANK(OPEN-TSMIN(OPEN,12)) < RANK(RANK(CORR(SUM((HIGH+LOW)/2,19), SUM(MEAN(VOLUME,40),19), 13))^5)
    part1 = RANK(open - TSMIN(open, 12))
    corr_val = CORR(SUM((high + low) / 2, 19), SUM(MEAN(vol, 40), 19), 13)
    part2 = RANK(RANK(corr_val) ** 5)
    return (part1 < part2).astype(float)


def alpha57(open, high, low, close, vol, amount, vwap, **kw):
    # [A] SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100, 3, 1)
    range9 = TSMAX(high, 9) - TSMIN(low, 9)
    with np.errstate(invalid='ignore', divide='ignore'):
        val = np.where(range9 > 0, (close - TSMIN(low, 9)) / range9 * 100, 0)
    return SMA(val, 3, 1)


def alpha58(open, high, low, close, vol, amount, vwap, **kw):
    # [A] COUNT(CLOSE>DELAY(CLOSE,1),20)/20*100
    return COUNT(close > DELAY(close, 1), 20) / 20 * 100


def alpha59(open, high, low, close, vol, amount, vwap, **kw):
    # [A] SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),20)
    dc1 = DELAY(close, 1)
    val = np.where(close == dc1, 0,
                   np.where(close > dc1, close - np.minimum(low, dc1), close - np.maximum(high, dc1)))
    return SUM(val, 20)


def alpha60(open, high, low, close, vol, amount, vwap, **kw):
    # [A] SUM(((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW)*VOLUME, 20)
    with np.errstate(invalid='ignore', divide='ignore'):
        inner = ((close - low) - (high - close)) / (high - low) * vol
    return SUM(inner, 20)


def alpha61(open, high, low, close, vol, amount, vwap, **kw):
    # [A] MAX(RANK(DECAYLINEAR(DELTA(VWAP,1),12)), RANK(DECAYLINEAR(RANK(CORR(LOW,MEAN(VOLUME,80),8)),17))) * -1
    corr_val = CORR(low, MEAN(vol, 80), 8)
    return -1 * np.maximum(
        RANK(DECAYLINEAR(DELTA(vwap, 1), 12)),
        RANK(DECAYLINEAR(RANK(corr_val), 17))
    )


def alpha62(open, high, low, close, vol, amount, vwap, **kw):
    # [A] -1*CORR(HIGH, RANK(VOLUME), 5)
    return -1 * CORR(high, RANK(vol), 5)


def alpha63(open, high, low, close, vol, amount, vwap, **kw):
    # [A] SMA(MAX(CLOSE-DELAY(CLOSE,1),0),6,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),6,1)*100
    dc1 = DELAY(close, 1)
    up = np.maximum(close - dc1, 0)
    abs_val = ABS(close - dc1)
    with np.errstate(invalid='ignore', divide='ignore'):
        return SMA(up, 6, 1) / SMA(abs_val, 6, 1) * 100


def alpha64(open, high, low, close, vol, amount, vwap, **kw):
    # [B] MAX为逐元素取大
    # MAX(RANK(DECAYLINEAR(CORR(RANK(VWAP),RANK(VOLUME),4),4)), RANK(DECAYLINEAR(MAX(CORR(RANK(CLOSE),RANK(MEAN(VOLUME,60)),4),13),14))) * -1
    corr1 = CORR(RANK(vwap), RANK(vol), 4)
    corr2 = CORR(RANK(close), RANK(MEAN(vol, 60)), 4)
    return -1 * np.maximum(
        RANK(DECAYLINEAR(corr1, 4)),
        RANK(DECAYLINEAR(np.maximum(corr2, 0), 14))  # MAX(X,13)→TSMAX(X,13)
    )


def alpha65(open, high, low, close, vol, amount, vwap, **kw):
    # [A] MEAN(CLOSE,6)/CLOSE
    return MEAN(close, 6) / close


def alpha66(open, high, low, close, vol, amount, vwap, **kw):
    # [A] (CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)*100
    mc6 = MEAN(close, 6)
    return (close - mc6) / mc6 * 100


def alpha67(open, high, low, close, vol, amount, vwap, **kw):
    # [A] SMA(MAX(CLOSE-DELAY(CLOSE,1),0),24,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),24,1)*100
    dc1 = DELAY(close, 1)
    up = np.maximum(close - dc1, 0)
    abs_val = ABS(close - dc1)
    with np.errstate(invalid='ignore', divide='ignore'):
        return SMA(up, 24, 1) / SMA(abs_val, 24, 1) * 100


def alpha68(open, high, low, close, vol, amount, vwap, **kw):
    # [A] SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME, 15, 2)
    inner = ((high + low) / 2 - (DELAY(high, 1) + DELAY(low, 1)) / 2) * (high - low) / vol
    return SMA(inner, 15, 2)


def alpha69(open, high, low, close, vol, amount, vwap, **kw):
    # [B] DTM/DBM, 条件表达式
    from src.lib.special import DTM, DBM
    dtm = DTM(open, high, low, close, 20)
    dbm = DBM(open, high, low, close, 20)
    sum_dtm = SUM(dtm, 20)
    sum_dbm = SUM(dbm, 20)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(
            sum_dtm > sum_dbm, (sum_dtm - sum_dbm) / sum_dtm,
            np.where(sum_dtm == sum_dbm, 0, (sum_dtm - sum_dbm) / sum_dbm)
        )


def alpha70(open, high, low, close, vol, amount, vwap, **kw):
    # [A] STD(AMOUNT,6)
    return STD(amount, 6)


def alpha71(open, high, low, close, vol, amount, vwap, **kw):
    # [A] (CLOSE-MEAN(CLOSE,24))/MEAN(CLOSE,24)*100
    mc24 = MEAN(close, 24)
    return (close - mc24) / mc24 * 100


def alpha72(open, high, low, close, vol, amount, vwap, **kw):
    # [A] SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100, 15, 1)
    range6 = TSMAX(high, 6) - TSMIN(low, 6)
    with np.errstate(invalid='ignore', divide='ignore'):
        val = np.where(range6 > 0, (TSMAX(high, 6) - close) / range6 * 100, 0)
    return SMA(val, 15, 1)


def alpha73(open, high, low, close, vol, amount, vwap, **kw):
    # [A] (TSRANK(DECAYLINEAR(DECAYLINEAR(CORR(CLOSE,VOLUME,10),16),4),5) - RANK(DECAYLINEAR(CORR(VWAP,MEAN(VOLUME,30),4),3))) * -1
    corr1 = CORR(close, vol, 10)
    part1 = TSRANK(DECAYLINEAR(DECAYLINEAR(corr1, 16), 4), 5)
    corr2 = CORR(vwap, MEAN(vol, 30), 4)
    part2 = RANK(DECAYLINEAR(corr2, 3))
    return -1 * (part1 - part2)


def alpha74(open, high, low, close, vol, amount, vwap, **kw):
    # [A] RANK(CORR(SUM(LOW*0.35+VWAP*0.65,20), SUM(MEAN(VOLUME,40),20), 7)) + RANK(CORR(RANK(VWAP), RANK(VOLUME), 6))
    s1 = SUM(low * 0.35 + vwap * 0.65, 20)
    s2 = SUM(MEAN(vol, 40), 20)
    return RANK(CORR(s1, s2, 7)) + RANK(CORR(RANK(vwap), RANK(vol), 6))


def alpha75(open, high, low, close, vol, amount, vwap, benchmark_open=None, benchmark_close=None, **kw):
    # [B] 基准指数变量名反了: BANCHMARKINDEXCLOSE=开盘价, BANCHMARKINDEXOPEN=收盘价
    # 按语义: CLOSE>OPEN & 基准收盘<基准开盘 → 股跌
    if benchmark_open is None or benchmark_close is None:
        return np.full_like(close, np.nan)
    # 修正: BANCHMARKINDEXCLOSE实际是开盘价, BANCHMARKINDEXOPEN实际是收盘价
    bm_o = benchmark_open
    bm_c = benchmark_close
    cond1 = (close > open) & (bm_c < bm_o)
    cond2 = (bm_c < bm_o)
    return COUNT(cond1, 50) / COUNT(cond2, 50)


def alpha76(open, high, low, close, vol, amount, vwap, **kw):
    # [A] STD(ABS(CLOSE/DELAY(CLOSE,1)-1)/VOLUME, 20) / MEAN(ABS(CLOSE/DELAY(CLOSE,1)-1)/VOLUME, 20)
    dc1 = DELAY(close, 1)
    with np.errstate(invalid='ignore', divide='ignore'):
        val = ABS(close / dc1 - 1) / vol
    return STD(val, 20) / MEAN(val, 20)


def alpha77(open, high, low, close, vol, amount, vwap, **kw):
    # [A] MIN(RANK(DECAYLINEAR(((HIGH+LOW)/2+HIGH)-(VWAP+HIGH),20)), RANK(DECAYLINEAR(CORR((HIGH+LOW)/2, MEAN(VOLUME,40),3),6)))
    inner1 = (high + low) / 2 + high - (vwap + high)
    corr_val = CORR((high + low) / 2, MEAN(vol, 40), 3)
    return np.minimum(
        RANK(DECAYLINEAR(inner1, 20)),
        RANK(DECAYLINEAR(corr_val, 6))
    )


def alpha78(open, high, low, close, vol, amount, vwap, **kw):
    # [C] MA→MEAN
    hlc3 = (high + low + close) / 3
    ma12 = MEAN(hlc3, 12)
    with np.errstate(invalid='ignore', divide='ignore'):
        return (hlc3 - ma12) / (0.015 * MEAN(ABS(close - ma12), 12))


def alpha79(open, high, low, close, vol, amount, vwap, **kw):
    # [A] SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100
    dc1 = DELAY(close, 1)
    up = np.maximum(close - dc1, 0)
    abs_val = ABS(close - dc1)
    with np.errstate(invalid='ignore', divide='ignore'):
        return SMA(up, 12, 1) / SMA(abs_val, 12, 1) * 100


def alpha80(open, high, low, close, vol, amount, vwap, **kw):
    # [A] (VOLUME-DELAY(VOLUME,5))/DELAY(VOLUME,5)*100
    dv5 = DELAY(vol, 5)
    return (vol - dv5) / dv5 * 100


def alpha81(open, high, low, close, vol, amount, vwap, **kw):
    # [A] SMA(VOLUME,21,2)
    return SMA(vol, 21, 2)


def alpha82(open, high, low, close, vol, amount, vwap, **kw):
    # [A] SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100, 20, 1)
    range6 = TSMAX(high, 6) - TSMIN(low, 6)
    with np.errstate(invalid='ignore', divide='ignore'):
        val = np.where(range6 > 0, (TSMAX(high, 6) - close) / range6 * 100, 0)
    return SMA(val, 20, 1)


def alpha83(open, high, low, close, vol, amount, vwap, **kw):
    # [A] -1*RANK(COVIANCE(RANK(HIGH), RANK(VOLUME), 5))
    return -1 * RANK(COVIANCE(RANK(high), RANK(vol), 5))


def alpha84(open, high, low, close, vol, amount, vwap, **kw):
    # [A] SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)), 20)
    dc1 = DELAY(close, 1)
    val = np.where(close > dc1, vol, np.where(close < dc1, -vol, 0.0))
    return SUM(val, 20)


def alpha85(open, high, low, close, vol, amount, vwap, **kw):
    # [A] TSRANK(VOLUME/MEAN(VOLUME,20), 20) * TSRANK(-1*DELTA(CLOSE,7), 8)
    return TSRANK(vol / MEAN(vol, 20), 20) * TSRANK(-1 * DELTA(close, 7), 8)


def alpha86(open, high, low, close, vol, amount, vwap, **kw):
    # [A] 条件表达式
    dc20 = DELAY(close, 20)
    dc10 = DELAY(close, 10)
    inner = (dc20 - dc10) / 10 - (dc10 - close) / 10
    return np.where(
        0.25 < inner, -1.0,
        np.where(inner < 0, 1.0, -1.0 * (close - DELAY(close, 1)))
    )


def alpha87(open, high, low, close, vol, amount, vwap, **kw):
    # [D] LOW*0.9+LOW*0.1=LOW
    inner = (low - vwap) / (open - (high + low) / 2)
    return -1 * (RANK(DECAYLINEAR(DELTA(vwap, 4), 7)) + TSRANK(DECAYLINEAR(inner, 11), 7))


def alpha88(open, high, low, close, vol, amount, vwap, **kw):
    # [A] (CLOSE-DELAY(CLOSE,20))/DELAY(CLOSE,20)*100
    dc20 = DELAY(close, 20)
    return (close - dc20) / dc20 * 100


def alpha89(open, high, low, close, vol, amount, vwap, **kw):
    # [A] 2*(SMA(CLOSE,13,2)-SMA(CLOSE,27,2)-SMA(SMA(CLOSE,13,2)-SMA(CLOSE,27,2),10,2))
    sma13 = SMA(close, 13, 2)
    sma27 = SMA(close, 27, 2)
    diff = sma13 - sma27
    return 2 * (diff - SMA(diff, 10, 2))


def alpha90(open, high, low, close, vol, amount, vwap, **kw):
    # [A] RANK(CORR(RANK(VWAP), RANK(VOLUME), 5)) * -1
    return -1 * RANK(CORR(RANK(vwap), RANK(vol), 5))


def alpha91(open, high, low, close, vol, amount, vwap, **kw):
    # [B] MAX为逐元素取大
    # RANK(CLOSE-MAX(CLOSE,5)) * RANK(CORR(MEAN(VOLUME,40), LOW, 5)) * -1
    return -1 * RANK(close - np.maximum(close, 0)) * RANK(CORR(MEAN(vol, 40), low, 5))


def alpha92(open, high, low, close, vol, amount, vwap, **kw):
    # [A] MAX(RANK(DECAYLINEAR(DELTA(CLOSE*0.35+VWAP*0.65,2),3)), TSRANK(DECAYLINEAR(ABS(CORR(MEAN(VOLUME,180),CLOSE,13)),5),15)) * -1
    inner1 = DELTA(close * 0.35 + vwap * 0.65, 2)
    corr_val = CORR(MEAN(vol, 180), close, 13)
    return -1 * np.maximum(
        RANK(DECAYLINEAR(inner1, 3)),
        TSRANK(DECAYLINEAR(ABS(corr_val), 5), 15)
    )


def alpha93(open, high, low, close, vol, amount, vwap, **kw):
    # [B] DBM方向退化
    do1 = DELAY(open, 1)
    cond = open >= do1
    val = np.where(cond, 0, np.maximum(open - low, open - do1))
    return SUM(val, 20)


def alpha94(open, high, low, close, vol, amount, vwap, **kw):
    # [A] SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)), 30)
    dc1 = DELAY(close, 1)
    val = np.where(close > dc1, vol, np.where(close < dc1, -vol, 0.0))
    return SUM(val, 30)


def alpha95(open, high, low, close, vol, amount, vwap, **kw):
    # [A] STD(AMOUNT, 20)
    return STD(amount, 20)


def alpha96(open, high, low, close, vol, amount, vwap, **kw):
    # [A] SMA(SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1),3,1)
    range9 = TSMAX(high, 9) - TSMIN(low, 9)
    with np.errstate(invalid='ignore', divide='ignore'):
        val = np.where(range9 > 0, (close - TSMIN(low, 9)) / range9 * 100, 0)
    return SMA(SMA(val, 3, 1), 3, 1)


def alpha97(open, high, low, close, vol, amount, vwap, **kw):
    # [A] STD(VOLUME, 10)
    return STD(vol, 10)


def alpha98(open, high, low, close, vol, amount, vwap, **kw):
    # [A] 条件表达式
    s100 = SUM(close, 100) / 100
    dc100 = DELAY(close, 100)
    with np.errstate(invalid='ignore', divide='ignore'):
        ratio = DELTA(s100, 100) / dc100
    return np.where(
        (ratio < 0.05) | (ratio == 0.05), -1 * (close - TSMIN(close, 100)), -1 * DELTA(close, 3)
    )


def alpha99(open, high, low, close, vol, amount, vwap, **kw):
    # [A] -1*RANK(COVIANCE(RANK(CLOSE), RANK(VOLUME), 5))
    return -1 * RANK(COVIANCE(RANK(close), RANK(vol), 5))


def alpha100(open, high, low, close, vol, amount, vwap, **kw):
    # [A] STD(VOLUME, 20)
    return STD(vol, 20)
