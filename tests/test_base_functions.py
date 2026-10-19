"""基础函数单元测试"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.lib.ts_operators import (
    DELTA, DELAY, SUM, MEAN, STD, CORR, TSMAX, TSMIN,
    TSRANK, WMA, DECAYLINEAR, SMA, COUNT, PROD, HIGHDAY, LOWDAY,
    SUMIF, SUMAC, FILTER, ABS, LOG, SIGN, POWER, SCALE, WHERE
)
from src.lib.cross_section import RANK, REGBETA, REGRESI, SEQUENCE
from src.lib.special import DTM, DBM, SELF


def make_panel(T=20, N=5, seed=42):
    np.random.seed(seed)
    return np.random.randn(T, N)


class TestDeltaDelay:
    def test_delta(self):
        S = make_panel(10, 3)
        d = DELTA(S, 1)
        assert np.allclose(d[1:], S[1:] - S[:-1])
        assert np.isnan(d[0]).all()

    def test_delta_d3(self):
        S = make_panel(10, 3)
        d = DELTA(S, 3)
        assert np.allclose(d[3:], S[3:] - S[:-3])
        assert np.isnan(d[:3]).all()

    def test_delay(self):
        S = make_panel(10, 3)
        d = DELAY(S, 2)
        assert np.allclose(d[2:], S[:-2])
        assert np.isnan(d[:2]).all()

    def test_delay_zero(self):
        S = make_panel(10, 3)
        d = DELAY(S, 0)
        assert np.allclose(d, S)


class TestRolling:
    def test_sum(self):
        S = make_panel(10, 3)
        s = SUM(S, 3)
        assert np.allclose(s[2], S[0:3].sum(axis=0))
        assert np.isnan(s[1]).all()

    def test_mean(self):
        S = make_panel(10, 3)
        m = MEAN(S, 5)
        assert np.allclose(m[4], S[0:5].mean(axis=0))
        assert np.isnan(m[3]).all()

    def test_std(self):
        S = make_panel(10, 3)
        s = STD(S, 5, ddof=1)
        assert np.allclose(s[4], np.std(S[0:5], axis=0, ddof=1))

    def test_std_zero_var(self):
        S = np.ones((10, 3)) * 5.0
        s = STD(S, 5, ddof=1)
        assert np.all(s[4] == 0.0)

    def test_corr(self):
        S1 = make_panel(20, 3)
        S2 = make_panel(20, 3, seed=99)
        c = CORR(S1, S2, 10)
        for j in range(3):
            expected = np.corrcoef(S1[0:10, j], S2[0:10, j])[0, 1]
            assert np.isclose(c[9, j], expected)

    def test_corr_zero_var(self):
        S1 = make_panel(10, 3)
        S2 = np.ones((10, 3))
        c = CORR(S1, S2, 5)
        assert np.all(c[4] == 0.0)

    def test_tsmax(self):
        S = make_panel(10, 3)
        m = TSMAX(S, 3)
        assert np.allclose(m[2], S[0:3].max(axis=0))

    def test_tsmin(self):
        S = make_panel(10, 3)
        m = TSMIN(S, 3)
        assert np.allclose(m[2], S[0:3].min(axis=0))


class TestTSRank:
    def test_basic(self):
        S = make_panel(10, 3)
        r = TSRANK(S, 5)
        assert r.shape == S.shape
        assert np.isnan(r[3]).all()
        # 当日是窗口最大值时，排名应为1.0
        S2 = S.copy()
        S2[4] = S2[0:5].max(axis=0) + 1
        r2 = TSRANK(S2, 5)
        assert np.allclose(r2[4], 1.0)

    def test_range(self):
        S = make_panel(50, 10)
        r = TSRANK(S, 20)
        valid = r[19:]
        assert np.all(valid[~np.isnan(valid)] >= 0)
        assert np.all(valid[~np.isnan(valid)] <= 1)


class TestWeighted:
    def test_wma(self):
        S = make_panel(10, 3)
        w = WMA(S, 3)
        # 权重 3,2,1 归一化
        wgt = np.array([3, 2, 1], dtype=float)
        wgt = wgt / wgt.sum()
        expected = (S[0:3] * wgt[:, None]).sum(axis=0)
        assert np.allclose(w[2], expected)

    def test_decaylinear_equals_wma(self):
        S = make_panel(10, 3)
        d = DECAYLINEAR(S, 5)
        w = WMA(S, 5)
        # 忽略NaN比较
        mask = ~(np.isnan(d) | np.isnan(w))
        assert np.allclose(d[mask], w[mask])


class TestSMA:
    def test_recursion(self):
        S = make_panel(10, 3)
        sma = SMA(S, 10, 2)
        # 第一天 = 原值
        assert np.allclose(sma[0], S[0])
        # 第二天 = (2/10)*S[1] + (8/10)*sma[0]
        expected = (2.0/10) * S[1] + (8.0/10) * sma[0]
        assert np.allclose(sma[1], expected)

    def test_nan_skip(self):
        S = make_panel(10, 3)
        S[0, 0] = np.nan
        sma = SMA(S, 10, 2)
        # 第一个有效值在第1天
        assert np.isclose(sma[1, 0], S[1, 0])


class TestMisc:
    def test_count(self):
        cond = np.array([True, False, True, True, False])
        cond = cond[:, None].repeat(2, axis=1).astype(float)
        c = COUNT(cond, 3)
        assert c[2, 0] == 2

    def test_prod(self):
        S = make_panel(10, 3)
        p = PROD(S, 3)
        assert np.allclose(p[2], S[0] * S[1] * S[2])

    def test_highday(self):
        S = np.zeros((10, 1))
        S[5, 0] = 100  # 最大值在第5天（窗口内）
        hd = HIGHDAY(S, 10)
        assert hd[9, 0] == 4  # 距今4天

    def test_lowday(self):
        S = np.zeros((10, 1))
        S[3, 0] = -100  # 最小值在第3天
        ld = LOWDAY(S, 10)
        assert ld[9, 0] == 6  # 距今6天

    def test_sumif(self):
        S = np.ones((10, 1))
        cond = np.array([True, False, True, False, True, False, True, False, True, False])
        cond = cond[:, None].astype(float)
        s = SUMIF(S, cond, 5)
        assert s[4, 0] == 3  # 5日内3个True

    def test_filter(self):
        cond = np.zeros((10, 1), dtype=bool)
        cond[0, 0] = True
        f = FILTER(cond, 3)
        assert f[0, 0] == True
        assert f[1, 0] == False
        assert f[2, 0] == False

    def test_log(self):
        S = np.array([[1.0, np.e, -1.0]])
        l = LOG(S)
        assert np.isclose(l[0, 0], 0)
        assert np.isclose(l[0, 1], 1)
        assert np.isnan(l[0, 2])


class TestCrossSection:
    def test_rank(self):
        S = np.array([[3.0, 1.0, 2.0, np.nan]])
        r = RANK(S)
        assert r.shape == (1, 4)
        assert np.isclose(r[0, 0], 1.0)  # 最大值 → 1.0
        assert np.isclose(r[0, 1], 0.0)  # 最小值 → 0.0
        assert np.isclose(r[0, 2], 0.5)  # 中间 → 0.5

    def test_rank_ties(self):
        S = np.array([[1.0, 1.0, 3.0]])
        r = RANK(S)
        # 两个并列最小值，平均排名=(0+1)/2=0.5，归一化0.5/(3-1)=0.25
        assert np.isclose(r[0, 0], 0.25)
        assert np.isclose(r[0, 1], 0.25)
        assert np.isclose(r[0, 2], 1.0)

    def test_regbeta(self):
        T, N = 20, 3
        x = np.arange(1, T + 1, dtype=float)
        S = np.zeros((T, N))
        for j in range(N):
            S[:, j] = 2.0 * x + j + np.random.randn(T) * 0.01
        beta = REGBETA(S, x, T)
        assert np.allclose(beta[-1], 2.0, atol=0.05)

    def test_regresi(self):
        T, N = 20, 3
        x = np.arange(1, T + 1, dtype=float)
        S = np.zeros((T, N))
        for j in range(N):
            S[:, j] = 2.0 * x + 1.0
        resi = REGRESI(S, x, T)
        # 完美线性拟合，残差应≈0
        assert np.allclose(resi[-1], 0, atol=0.01)

    def test_sequence(self):
        s = SEQUENCE(5)
        assert np.array_equal(s, [1, 2, 3, 4, 5])


class TestSpecial:
    def test_dtm(self):
        T, N = 10, 3
        OPEN = np.random.uniform(10, 20, (T, N))
        HIGH = OPEN + np.random.uniform(0, 2, (T, N))
        LOW = OPEN - np.random.uniform(0, 2, (T, N))
        CLOSE = np.random.uniform(10, 20, (T, N))
        d = DTM(OPEN, HIGH, LOW, CLOSE, 5)
        assert d.shape == (T, N)

    def test_dbm(self):
        T, N = 10, 3
        OPEN = np.random.uniform(10, 20, (T, N))
        HIGH = OPEN + np.random.uniform(0, 2, (T, N))
        LOW = OPEN - np.random.uniform(0, 2, (T, N))
        CLOSE = np.random.uniform(10, 20, (T, N))
        d = DBM(OPEN, HIGH, LOW, CLOSE, 5)
        assert d.shape == (T, N)

    def test_self(self):
        S = make_panel(10, 3)
        s = SELF(S, 0, init_value=0.5)
        assert s[0, 0] == 0.5
        assert np.allclose(s[1:], S[:-1])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
