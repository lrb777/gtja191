# coding=utf-8
"""
Step2 测试版：只算前10个因子，测速
"""
import sys, os, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from config.settings import DATA_PROCESSED, OUTPUT, PREDICTION_HORIZON
from src.factors.alpha_001_050 import *

panel = np.load(DATA_PROCESSED / "panel.npz")
T, N = panel["close"].shape
data = {k: panel[k] for k in panel.files}

# 基准
bm_ret = np.nanmean(data["ret"], axis=1, keepdims=True)
bm_close = np.cumprod(1 + bm_ret, axis=0)
bm_open = np.roll(bm_close, 1); bm_open[0] = bm_close[0]
data["benchmark_close"] = np.tile(bm_close, (1, N))
data["benchmark_open"] = np.tile(bm_open, (1, N))

funcs = [alpha01, alpha02, alpha03, alpha04, alpha05, alpha06, alpha07, alpha08, alpha09, alpha11]

for f in funcs:
    t0 = time.time()
    try:
        result = f(**data)
        dt = time.time() - t0
        print(f"{f.__name__}: {dt:.1f}s shape={result.shape} nan={np.isnan(result).mean():.1%}")
    except Exception as e:
        dt = time.time() - t0
        print(f"{f.__name__}: {dt:.1f}s ERROR {e}")
