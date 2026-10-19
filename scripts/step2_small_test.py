# coding=utf-8
"""小样本测试：只取100只股票"""
import sys, os, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, r'D:\gtja191')

import numpy as np
from config.settings import DATA_PROCESSED, OUTPUT, PREDICTION_HORIZON

panel = np.load(DATA_PROCESSED / "panel.npz")
T, N = panel["close"].shape
N_sub = 100  # 只取100只

data = {k: panel[k][:, :N_sub] for k in panel.files}
print(f"T={T}, N_sub={N_sub}")

bm_ret = np.nanmean(data["ret"], axis=1, keepdims=True)
bm_close = np.cumprod(1 + bm_ret, axis=0)
bm_open = np.roll(bm_close, 1); bm_open[0] = bm_close[0]
data["benchmark_close"] = np.tile(bm_close, (1, N_sub))
data["benchmark_open"] = np.tile(bm_open, (1, N_sub))

h = PREDICTION_HORIZON
fwd_ret = np.full_like(data["close"], np.nan)
fwd_ret[:-h] = data["close"][h:] / data["close"][:-h] - 1

from src.factors.alpha_001_050 import *
from src.factors.alpha_051_100 import *
from src.factors.alpha_101_150 import *
from src.factors.alpha_151_191 import *

all_funcs = {k: v for k, v in sorted(globals().items()) if k.startswith('alpha') and callable(v)}
print(f"共 {len(all_funcs)} 个因子")

ok = 0
fail = 0
for name, func in all_funcs.items():
    t0 = time.time()
    try:
        result = func(**data)
        dt = time.time() - t0
        if dt > 5:
            print(f"  SLOW: {name} {dt:.1f}s")
        ok += 1
    except Exception as e:
        print(f"  ERROR: {name} {e}")
        fail += 1

print(f"\nOK: {ok}, FAIL: {fail}")
