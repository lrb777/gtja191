# coding=utf-8
"""
Alpha191 因子实现 — 第1批 (Alpha1~Alpha50)

分类标记：
  [A] = 无专属缺项
  [B] = 语义歧义
  [C] = 因子专属缺项
  [D] = 代数冗余/尺度敏感

输入数据约定（全部前复权，形状 (T, N) = 交易日×股票）：
  open, high, low, close, vol, amount, vwap
  ret = close/DELAY(close,1) - 1
  benchmark_open, benchmark_close  — 中证500
"""
import numpy as np
from src.lib.ts_operators import *
from src.lib.cross_section import RANK, REGBETA, REGRESI, SEQUENCE


def alpha01(open, high, low, close, vol, amount, vwap, **kw):
    # [A] -1 * CORR(RANK(DELTA(LOG(VOLUME), 1)), RANK((CLOSE-OPEN)/OPEN), 6)
    log_vol = LOG(vol.astype(float))
    delta_log_vol = DELTA(log_vol, 1)
    rank_delta = RANK(delta_log_vol)
    rank_co = RANK((close - open) / open)
    return -1 * CORR(rank_delta, rank_co, 6)


def alpha02(open, high, low, close, vol, amount, vwap, **kw):
    # [A] -1 * DELTA(((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW), 1)
    inner = ((close - low) - (high - close)) / (high - low)
    return -1 * DELTA(inner, 1)


def alpha03(open, high, low, close, vol, amount, vwap, **kw):
    # [A] SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),6)
    dc = DELAY(close, 1)
    cond_eq = (close == dc)
    cond_up = (close > dc)
    val = np.where(cond_eq, 0,
                   np.where(cond_up, close - np.minimum(low, dc), close - np.maximum(high, dc)))
    return SUM(val, 6)


def alpha04(open, high, low, close, vol, amount, vwap, **kw):
    # [A] 复杂条件表达式
    s8 = SUM(close, 8) / 8
    std8 = STD(close, 8)
    s2 = SUM(close, 2) / 2
    mv20 = MEAN(vol, 20)
    vr = vol / mv20
    result = np.where(
        (s8 + std8) < s2, -1.0,
        np.where(
            s2 < (s8 - std8), 1.0,
            np.where((vr > 1) | (vr == 1), 1.0, -1.0)
        )
    )
    return result


def alpha05(open, high, low, close, vol, amount, vwap, **kw):
    # [A] -1 * TSMAX(CORR(TSRANK(VOLUME,5), TSRANK(HIGH,5), 5), 3)
    tsr_vol = TSRANK(vol, 5)
    tsr_high = TSRANK(high, 5)
    corr_val = CORR(tsr_vol, tsr_high, 5)
    return -1 * TSMAX(corr_val, 3)


def alpha06(open, high, low, close, vol, amount, vwap, **kw):
    # [A] RANK(SIGN(DELTA((OPEN*0.85+HIGH*0.15), 4))) * -1
    inner = open * 0.85 + high * 0.15
    return -1 * RANK(SIGN(DELTA(inner, 4)))


def alpha07(open, high, low, close, vol, amount, vwap, **kw):
    # [B] MAX/MIN逐元素取大/小
    # (RANK(MAX((VWAP-CLOSE),3)) + RANK(MIN((VWAP-CLOSE),3))) * RANK(DELTA(VOLUME,3))
    # 按分类B: MAX(X,3)为滚动3日最大值
    diff = vwap - close
    return (RANK(TSMAX(diff, 3)) + RANK(TSMIN(diff, 3))) * RANK(DELTA(vol, 3))


def alpha08(open, high, low, close, vol, amount, vwap, **kw):
    # [A] RANK(DELTA(((HIGH+LOW)/2*0.2 + VWAP*0.8), 4)) * -1
    inner = (high + low) / 2 * 0.2 + vwap * 0.8
    return -1 * RANK(DELTA(inner, 4))


def alpha09(open, high, low, close, vol, amount, vwap, **kw):
    # [A] SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME, 7, 2)
    inner = ((high + low) / 2 - (DELAY(high, 1) + DELAY(low, 1)) / 2) * (high - low) / vol
    return SMA(inner, 7, 2)


def alpha10(open, high, low, close, vol, amount, vwap, ret=None, **kw):
    # [C] RANK(MAX(((RET<0)?STD(RET,20):CLOSE)^2), 5)
    # 审计: MAX只有一个参数, RANK多出参数5
    # 候选修正: RANK(TSMAX(WHERE(RET<0, STD(RET,20), CLOSE)^2, 5))
    if ret is None:
        ret = close / DELAY(close, 1) - 1
    val = np.where(ret < 0, STD(ret, 20), close)
    return RANK(TSMAX(val ** 2, 5))


def alpha11(open, high, low, close, vol, amount, vwap, **kw):
    # [A] SUM(((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW)*VOLUME, 6)
    inner = ((close - low) - (high - close)) / (high - low) * vol
    return SUM(inner, 6)


def alpha12(open, high, low, close, vol, amount, vwap, **kw):
    # [A] RANK(OPEN - SUM(VWAP,10)/10) * (-1*RANK(ABS(CLOSE-VWAP)))
    return RANK(open - SUM(vwap, 10) / 10) * (-1 * RANK(ABS(close - vwap)))


def alpha13(open, high, low, close, vol, amount, vwap, **kw):
    # [A] (HIGH*LOW)^0.5 - VWAP
    return np.sqrt(high * low) - vwap


def alpha14(open, high, low, close, vol, amount, vwap, **kw):
    # [A] CLOSE - DELAY(CLOSE, 5)
    return close - DELAY(close, 5)


def alpha15(open, high, low, close, vol, amount, vwap, **kw):
    # [A] OPEN/DELAY(CLOSE,1) - 1
    return open / DELAY(close, 1) - 1


def alpha16(open, high, low, close, vol, amount, vwap, **kw):
    # [A] -1 * TSMAX(RANK(CORR(RANK(VOLUME), RANK(VWAP), 5)), 5)
    corr_val = CORR(RANK(vol), RANK(vwap), 5)
    return -1 * TSMAX(RANK(corr_val), 5)


def alpha17(open, high, low, close, vol, amount, vwap, **kw):
    # [B] RANK((VWAP - MAX(VWAP, 15)))^DELTA(CLOSE, 5)
    # MAX(VWAP,15) → TSMAX(VWAP,15) 滚动
    inner = vwap - TSMAX(vwap, 15)
    return RANK(inner) ** DELTA(close, 5)


def alpha18(open, high, low, close, vol, amount, vwap, **kw):
    # [A] CLOSE/DELAY(CLOSE,5)
    return close / DELAY(close, 5)


def alpha19(open, high, low, close, vol, amount, vwap, **kw):
    # [A] 条件表达式
    dc5 = DELAY(close, 5)
    return np.where(
        close < dc5, (close - dc5) / dc5,
        np.where(close == dc5, 0, (close - dc5) / close)
    )


def alpha20(open, high, low, close, vol, amount, vwap, **kw):
    # [A] (CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100
    dc6 = DELAY(close, 6)
    return (close - dc6) / dc6 * 100


def alpha21(open, high, low, close, vol, amount, vwap, **kw):
    # [C] REGBETA(MEAN(CLOSE,6), SEQUENCE(6)) — 缺回归窗口
    # 候选: REGBETA(MEAN(CLOSE,6), SEQUENCE(6), 6)
    mc6 = MEAN(close, 6)
    T = close.shape[0]
    # SEQUENCE(6) → 重复到长度T, 窗口内为1~6
    seq = np.tile(SEQUENCE(6), (T // 6 + 1,))[:T]
    return REGBETA(mc6, seq, 6)


def alpha22(open, high, low, close, vol, amount, vwap, **kw):
    # [C] SMEAN→SMA, 补n=12,m=1
    # SMA(((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6) - DELAY((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6),3)), 12, 1)
    inner = (close - MEAN(close, 6)) / MEAN(close, 6)
    val = inner - DELAY(inner, 3)
    return SMA(val, 12, 1)


def alpha23(open, high, low, close, vol, amount, vwap, **kw):
    # [C] 表达式损坏, 候选: STD(CLOSE,20)在CLOSE>DELAY(CLOSE,1)时取值否则0
    dc1 = DELAY(close, 1)
    std20 = STD(close, 20)
    up_val = np.where(close > dc1, std20, 0.0)
    dn_val = np.where(close <= dc1, std20, 0.0)
    sma_up = SMA(up_val, 20, 1)
    sma_dn = SMA(dn_val, 20, 1)
    with np.errstate(invalid='ignore', divide='ignore'):
        return sma_up / (sma_up + sma_dn) * 100

                                                                                                                                      
def alpha24(open, high, low, close, vol, amount, vwap, **kw):
    # [A] SMA(CLOSE-DELAY(CLOSE,5), 5, 1)
    return SMA(close - DELAY(close, 5), 5, 1)


def alpha25(open, high, low, close, vol, amount, vwap, ret=None, **kw):
    # [A] -1*RANK(DELTA(CLOSE,7)*(1-RANK(DECAYLINEAR(VOLUME/MEAN(VOLUME,20),9))))*(1+RANK(SUM(RET,250)))
    if ret is None:
        ret = close / DELAY(close, 1) - 1
    inner1 = DELTA(close, 7) * (1 - RANK(DECAYLINEAR(vol / MEAN(vol, 20), 9)))
    return -1 * RANK(inner1) * (1 + RANK(SUM(ret, 250)))


def alpha26(open, high, low, close, vol, amount, vwap, **kw):
    # [D] (SUM(CLOSE,7)/7 - CLOSE) + CORR(VWAP, DELAY(CLOSE,5), 230)
    return (SUM(close, 7) / 7 - close) + CORR(vwap, DELAY(close, 5), 230)


def alpha27(open, high, low, close, vol, amount, vwap, **kw):
    # [A] WMA((CLOSE-DELAY(CLOSE,3))/DELAY(CLOSE,3)*100 + (CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100, 12)
    dc3 = DELAY(close, 3)
    dc6 = DELAY(close, 6)
    inner = (close - dc3) / dc3 * 100 + (close - dc6) / dc6 * 100
    return WMA(inner, 12)


def alpha28(open, high, low, close, vol, amount, vwap, **kw):
    # [B] 第二项分母MAX(HIGH,9)-TSMAX(LOW,9), 候选同第一项
    # 3*SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1)
    # -2*SMA(SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1),3,1)
    range9 = TSMAX(high, 9) - TSMIN(low, 9)
    with np.errstate(invalid='ignore', divide='ignore'):
        rsi_val = np.where(range9 > 0, (close - TSMIN(low, 9)) / range9 * 100, 0)
    sma1 = SMA(rsi_val, 3, 1)
    sma2 = SMA(sma1, 3, 1)
    return 3 * sma1 - 2 * sma2


def alpha29(open, high, low, close, vol, amount, vwap, **kw):
    # [A] (CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*VOLUME
    dc6 = DELAY(close, 6)
    return (close - dc6) / dc6 * vol


def alpha30(open, high, low, close, vol, amount, vwap, mkt=None, smb=None, hml=None, **kw):
    # [C] WMA((REGRESI(CLOSE/DELAY(CLOSE,1)-1, MKT, SMB, HML, 60))^2, 20)
    # 需三因子, 暂用None标记
    if mkt is None or smb is None or hml is None:
        return np.full_like(close, np.nan)
    dc1 = DELAY(close, 1)
    ret = close / dc1 - 1
    # 多元回归残差（需单独实现）
    # 暂用MARKER
    return np.full_like(close, np.nan)


def alpha31(open, high, low, close, vol, amount, vwap, **kw):
    # [A] (CLOSE-MEAN(CLOSE,12))/MEAN(CLOSE,12)*100
    mc12 = MEAN(close, 12)
    return (close - mc12) / mc12 * 100


def alpha32(open, high, low, close, vol, amount, vwap, **kw):
    # [A] -1 * SUM(RANK(CORR(RANK(HIGH), RANK(VOLUME), 3)), 3)
    return -1 * SUM(RANK(CORR(RANK(high), RANK(vol), 3)), 3)


def alpha33(open, high, low, close, vol, amount, vwap, ret=None, **kw):
    # [A] (-TSMIN(LOW,5)+DELAY(TSMIN(LOW,5),5)) * RANK((SUM(RET,240)-SUM(RET,20))/220) * TSRANK(VOLUME,5)
    if ret is None:
        ret = close / DELAY(close, 1) - 1
    part1 = -1 * TSMIN(low, 5) + DELAY(TSMIN(low, 5), 5)
    part2 = RANK((SUM(ret, 240) - SUM(ret, 20)) / 220)
    part3 = TSRANK(vol, 5)
    return part1 * part2 * part3


def alpha34(open, high, low, close, vol, amount, vwap, **kw):
    # [A] MEAN(CLOSE,12)/CLOSE
    return MEAN(close, 12) / close


def alpha35(open, high, low, close, vol, amount, vwap, **kw):
    # [D] MIN(RANK(DECAYLINEAR(DELTA(OPEN,1),15)), RANK(DECAYLINEAR(CORR(VOLUME, OPEN, 17),7))) * -1
    # 注: 原文OPEN*0.65+OPEN*0.35=OPEN
    corr_val = CORR(vol, open, 17)
    return -1 * np.minimum(
        RANK(DECAYLINEAR(DELTA(open, 1), 15)),
        RANK(DECAYLINEAR(corr_val, 7))
    )


def alpha36(open, high, low, close, vol, amount, vwap, **kw):
    # [C] RANK(SUM(CORR(RANK(VOLUME), RANK(VWAP), 6), 2))
    corr_val = CORR(RANK(vol), RANK(vwap), 6)
    return RANK(SUM(corr_val, 2))


def alpha37(open, high, low, close, vol, amount, vwap, ret=None, **kw):
    # [A] -1*RANK(SUM(OPEN,5)*SUM(RET,5) - DELAY(SUM(OPEN,5)*SUM(RET,5),10))
    if ret is None:
        ret = close / DELAY(close, 1) - 1
    inner = SUM(open, 5) * SUM(ret, 5)
    return -1 * RANK(inner - DELAY(inner, 10))


def alpha38(open, high, low, close, vol, amount, vwap, **kw):
    # [A] (SUM(HIGH,20)/20 < HIGH) ? -1*DELTA(HIGH,2) : 0
    return np.where(SUM(high, 20) / 20 < high, -1 * DELTA(high, 2), 0.0)


def alpha39(open, high, low, close, vol, amount, vwap, **kw):
    # [A] (RANK(DECAYLINEAR(DELTA(CLOSE,2),8)) - RANK(DECAYLINEAR(CORR(VWAP*0.3+OPEN*0.7, SUM(MEAN(VOLUME,180),37), 14), 12))) * -1
    corr_val = CORR(vwap * 0.3 + open * 0.7, SUM(MEAN(vol, 180), 37), 14)
    return -1 * (RANK(DECAYLINEAR(DELTA(close, 2), 8)) - RANK(DECAYLINEAR(corr_val, 12)))


def alpha40(open, high, low, close, vol, amount, vwap, **kw):
    # [A] SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:0),26) / SUM((CLOSE<=DELAY(CLOSE,1)?VOLUME:0),26) * 100
    dc1 = DELAY(close, 1)
    up = np.where(close > dc1, vol, 0.0)
    dn = np.where(close <= dc1, vol, 0.0)
    sum_up = SUM(up, 26)
    sum_dn = SUM(dn, 26)
    with np.errstate(invalid='ignore', divide='ignore'):
        return sum_up / sum_dn * 100


def alpha41(open, high, low, close, vol, amount, vwap, **kw):
    # [B] RANK(MAX(DELTA(VWAP,3),5)) * -1
    # MAX(X,5)→TSMAX(X,5)滚动
    return -1 * RANK(TSMAX(DELTA(vwap, 3), 5))


def alpha42(open, high, low, close, vol, amount, vwap, **kw):
    # [A] -1*RANK(STD(HIGH,10)) * CORR(HIGH, VOLUME, 10)
    return -1 * RANK(STD(high, 10)) * CORR(high, vol, 10)


def alpha43(open, high, low, close, vol, amount, vwap, **kw):
    # [A] SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)), 6)
    dc1 = DELAY(close, 1)
    val = np.where(close > dc1, vol, np.where(close < dc1, -vol, 0.0))
    return SUM(val, 6)


def alpha44(open, high, low, close, vol, amount, vwap, **kw):
    # [A] TSRANK(DECAYLINEAR(CORR(LOW, MEAN(VOLUME,10), 7), 6), 4) + TSRANK(DECAYLINEAR(DELTA(VWAP,3), 10), 15)
    part1 = TSRANK(DECAYLINEAR(CORR(low, MEAN(vol, 10), 7), 6), 4)
    part2 = TSRANK(DECAYLINEAR(DELTA(vwap, 3), 10), 15)
    return part1 + part2


def alpha45(open, high, low, close, vol, amount, vwap, **kw):
    # [A] RANK(DELTA(CLOSE*0.6+OPEN*0.4, 1)) * RANK(CORR(VWAP, MEAN(VOLUME,150), 15))
    inner = close * 0.6 + open * 0.4
    return RANK(DELTA(inner, 1)) * RANK(CORR(vwap, MEAN(vol, 150), 15))


def alpha46(open, high, low, close, vol, amount, vwap, **kw):
    # [A] (MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/(4*CLOSE)
    return (MEAN(close, 3) + MEAN(close, 6) + MEAN(close, 12) + MEAN(close, 24)) / (4 * close)


def alpha47(open, high, low, close, vol, amount, vwap, **kw):
    # [A] SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100, 9, 1)
    range6 = TSMAX(high, 6) - TSMIN(low, 6)
    with np.errstate(invalid='ignore', divide='ignore'):
        val = np.where(range6 > 0, (TSMAX(high, 6) - close) / range6 * 100, 0)
    return SMA(val, 9, 1)


def alpha48(open, high, low, close, vol, amount, vwap, **kw):
    # [A] -1*((RANK(SIGN(CLOSE-DELAY(CLOSE,1))+SIGN(DELAY(CLOSE,1)-DELAY(CLOSE,2))+SIGN(DELAY(CLOSE,2)-DELAY(CLOSE,3)))) * SUM(VOLUME,5)/SUM(VOLUME,20))
    s1 = SIGN(close - DELAY(close, 1))
    s2 = SIGN(DELAY(close, 1) - DELAY(close, 2))
    s3 = SIGN(DELAY(close, 2) - DELAY(close, 3))
    return -1 * RANK(s1 + s2 + s3) * SUM(vol, 5) / SUM(vol, 20)


def alpha49(open, high, low, close, vol, amount, vwap, **kw):
    # [D] 复杂SUM条件, 与Alpha50/51线性相关
    dh = DELAY(high, 1)
    dl = DELAY(low, 1)
    hl = high + low
    dhl = dh + dl
    up_cond = hl >= dhl
    dn_cond = hl <= dhl
    max_val = np.maximum(ABS(high - dh), ABS(low - dl))
    up_sum = SUM(np.where(up_cond, 0, max_val), 12)
    dn_sum = SUM(np.where(dn_cond, 0, max_val), 12)
    with np.errstate(invalid='ignore', divide='ignore'):
        return up_sum / (up_sum + dn_sum)


def alpha50(open, high, low, close, vol, amount, vwap, **kw):
    # [D] dn_sum/(dn_sum+up_sum) - up_sum/(up_sum+dn_sum)
    dh = DELAY(high, 1)
    dl = DELAY(low, 1)
    hl = high + low
    dhl = dh + dl
    up_cond = hl >= dhl
    dn_cond = hl <= dhl
    max_val = np.maximum(ABS(high - dh), ABS(low - dl))
    up_sum = SUM(np.where(up_cond, 0, max_val), 12)
    dn_sum = SUM(np.where(dn_cond, 0, max_val), 12)
    total = up_sum + dn_sum
    with np.errstate(invalid='ignore', divide='ignore'):
        return dn_sum / total - up_sum / total
