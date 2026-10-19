# coding=utf-8
"""
快速测试单个函数速度
"""
import sys, os, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, r'D:\gtja191')

import numpy as np
from config.settings import DATA_PROCESSED

panel = np.load(DATA_PROCESSED / "panel.npz")
T, N = panel["close"].shape
print(f"T={T}, N={N}")

close = panel["close"]
vol = panel["vol"]

# 测试 DELTA (简单)
from src.lib.ts_operators import DELTA, LOG, TSRANK, CORR, STD, SUM, MEAN
from src.lib.cross_section import RANK

t0 = time.time()
d = DELTA(close, 1)
print(f"DELTA: {time.time()-t0:.2f}s")

t0 = time.time()
l = LOG(vol.astype(float))
print(f"LOG: {time.time()-t0:.2f}s")

t0 = time.time()
r = RANK(close)
print(f"RANK: {time.time()-t0:.2f}s")

t0 = time.time()
s = SUM(close, 8)
print(f"SUM(8): {time.time()-t0:.2f}s")

t0 = time.time()
m = MEAN(close, 20)
print(f"MEAN(20): {time.time()-t0:.2f}s")

t0 = time.time()
st = STD(close, 20)
print(f"STD(20): {time.time()-t0:.2f}s")

t0 = time.time()
ts = TSRANK(close, 20)
print(f"TSRANK(20): {time.time()-t0:.2f}s")

t0 = time.time()
c = CORR(close, vol, 10)
print(f"CORR(10): {time.time()-t0:.2f}s")
