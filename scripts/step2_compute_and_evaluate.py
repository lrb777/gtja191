# coding=utf-8
"""
Step2: 191因子计算 + 三段三频IC分析（优化版v2）

优化：
1. 预计算前瞻收益排名（所有因子共享，只算一次）
2. 用pandas rank向量化替代逐日argsort循环
3. 每个频率只算一次IC，再按时间段切分统计（9次→3次）
"""
import sys, os, time, warnings, json
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from config.settings import DATA_PROCESSED, OUTPUT

from src.factors.alpha_001_050 import *
from src.factors.alpha_051_100 import *
from src.factors.alpha_101_150 import *
from src.factors.alpha_151_191 import *


def get_all_alpha_funcs():
    funcs = {}
    for name in sorted(globals().keys()):
        if name.startswith('alpha') and callable(globals()[name]):
            funcs[name] = globals()[name]
    return funcs


def precompute_ret_ranks(fwd_ret):
    """预计算前瞻收益的逐日截面排名和去中心化值（所有因子共享）"""
    T, N = fwd_ret.shape
    r_df = pd.DataFrame(fwd_ret)
    r_rank = r_df.rank(axis=1).values.astype(np.float64)
    
    mask = ~np.isnan(fwd_ret)
    r_rank = np.where(mask, r_rank, 0.0)
    
    n_valid = mask.sum(axis=1)
    r_mean = r_rank.sum(axis=1) / np.maximum(n_valid, 1)
    r_centered = r_rank - r_mean[:, None]
    r_centered = np.where(mask, r_centered, 0.0)
    r_norm = np.sqrt((r_centered ** 2).sum(axis=1))
    
    valid_day = (n_valid >= 20) & (r_norm > 0)
    return r_centered, r_norm, valid_day


def fast_rank_ic(factor, r_centered, r_norm, valid_day):
    """用预计算的收益排名，向量化计算每日IC"""
    T, N = factor.shape
    
    f_df = pd.DataFrame(factor)
    f_rank = f_df.rank(axis=1).values.astype(np.float64)
    
    mask = ~np.isnan(factor) & (r_norm > 0)[:, None]
    f_rank = np.where(mask, f_rank, 0.0)
    
    n_valid = mask.sum(axis=1)
    f_mean = f_rank.sum(axis=1) / np.maximum(n_valid, 1)
    f_centered = f_rank - f_mean[:, None]
    f_centered = np.where(mask, f_centered, 0.0)
    f_norm = np.sqrt((f_centered ** 2).sum(axis=1))
    
    num = (f_centered * r_centered).sum(axis=1)
    den = f_norm * r_norm
    ic = np.where(valid_day & (den > 0), num / np.where(den > 0, den, 1), np.nan)
    return ic[~np.isnan(ic)]


def compute_ic_stats(ic_arr):
    if len(ic_arr) < 10:
        return {'ic_mean': 0, 'ic_std': 0, 'ic_ir': 0, 't_value': 0, 'win_rate': 0, 'n': 0}
    ic_mean = ic_arr.mean()
    ic_std = ic_arr.std()
    ic_ir = ic_mean / ic_std if ic_std > 0 else 0
    t_val = ic_mean / (ic_std / np.sqrt(len(ic_arr))) if ic_std > 0 and len(ic_arr) > 1 else 0
    win_rate = (ic_arr > 0).mean() if ic_mean > 0 else (ic_arr < 0).mean()
    return {
        'ic_mean': float(ic_mean),
        'ic_std': float(ic_std),
        'ic_ir': float(ic_ir),
        't_value': float(t_val),
        'win_rate': float(win_rate),
        'n': int(len(ic_arr)),
    }


def main():
    print("加载面板数据...", flush=True)
    panel = np.load(DATA_PROCESSED / "panel.npz")
    dates = np.load(DATA_PROCESSED / "dates.npy", allow_pickle=True)
    T, N = panel["close"].shape
    print(f"  T={T}, N={N}", flush=True)

    data = {k: panel[k] for k in panel.files}
    close = data["close"]

    date_arr = pd.to_datetime(dates)
    seg1_mask = (date_arr < pd.Timestamp("2021-01-01"))
    seg2_mask = (date_arr >= pd.Timestamp("2021-01-01"))
    all_mask = np.ones(T, dtype=bool)

    horizons = [2, 5, 10]
    print("预计算前瞻收益排名...", flush=True)
    ret_ranks = {}
    for h in horizons:
        fr = np.full_like(close, np.nan)
        fr[:-h] = close[h:] / close[:-h] - 1
        r_c, r_n, v_d = precompute_ret_ranks(fr)
        ret_ranks[h] = (r_c, r_n, v_d)
        print(f"  h={h} done", flush=True)

    print("\n计算191因子 + IC...", flush=True)
    all_funcs = get_all_alpha_funcs()
    print(f"  共 {len(all_funcs)} 个因子", flush=True)

    results = {}
    n_ok = 0
    n_fail = 0
    t_start = time.time()

    for i, (name, func) in enumerate(all_funcs.items()):
        t0 = time.time()
        try:
            factor = func(**data)
            dt_factor = time.time() - t0

            if factor.shape != (T, N) or np.all(np.isnan(factor)):
                results[name] = {}
                n_fail += 1
                continue

            results[name] = {}
            for h in horizons:
                r_c, r_n, v_d = ret_ranks[h]
                ic_arr = fast_rank_ic_with_mask(factor, r_c, r_n, v_d)
                
                for seg_name, seg_mask in [('2015-2020', seg1_mask), ('2021-2026', seg2_mask), ('full', all_mask)]:
                    seg_ic = ic_arr[seg_mask]
                    seg_ic = seg_ic[~np.isnan(seg_ic)]
                    key = f"{seg_name}_h{h}"
                    results[name][key] = compute_ic_stats(seg_ic)

            n_ok += 1
            dt_total = time.time() - t0
            if dt_total > 10:
                print(f"  SLOW: {name} {dt_total:.1f}s (factor={dt_factor:.1f}s)", flush=True)
        except Exception as e:
            results[name] = {}
            n_fail += 1
            print(f"  ERROR: {name} {e}", flush=True)

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t_start
            eta = elapsed / (i + 1) * len(all_funcs)
            print(f"  进度: {i+1}/{len(all_funcs)} (ok={n_ok}, fail={n_fail}) elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)
        elif (i + 1) % 5 == 0:
            elapsed = time.time() - t_start
            print(f"  [{i+1}/{len(all_funcs)}] {name} {time.time()-t0:.1f}s elapsed={elapsed:.0f}s", flush=True)

    print(f"\n完成: ok={n_ok}, fail={n_fail}, 耗时={time.time()-t_start:.0f}s", flush=True)

    outpath = OUTPUT / "ic_results_3x3.json"
    with open(outpath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"IC结果已保存: {outpath}", flush=True)

    print("\n=== IC汇总（全期_2日，按|IC|降序前20）===", flush=True)
    sorted_factors = sorted(results.items(),
                           key=lambda x: abs(x[1].get('full_h2', {}).get('ic_mean', 0)),
                           reverse=True)
    print(f"{'因子':>10s} | {'15-20_h2':>10s} {'21-26_h2':>10s} {'full_h2':>10s} | {'full_h5':>10s} {'full_h10':>10s}")
    print("-" * 80)
    for name, segs in sorted_factors[:20]:
        v = segs.get
        ic1 = v('2015-2020_h2', {}).get('ic_mean', 0)
        ic2 = v('2021-2026_h2', {}).get('ic_mean', 0)
        ic3 = v('full_h2', {}).get('ic_mean', 0)
        ic5 = v('full_h5', {}).get('ic_mean', 0)
        ic10 = v('full_h10', {}).get('ic_mean', 0)
        print(f"{name:>10s} | {ic1:>+10.4f} {ic2:>+10.4f} {ic3:>+10.4f} | {ic5:>+10.4f} {ic10:>+10.4f}")

    print(f"\n=== 总体统计（全期_2日）===", flush=True)
    ic_means = [v.get('full_h2', {}).get('ic_mean', 0) for v in results.values()]
    t_vals = [abs(v.get('full_h2', {}).get('t_value', 0)) for v in results.values()]
    print(f"因子总数: {len(results)}")
    print(f"|IC|>0.02: {sum(1 for x in ic_means if abs(x) > 0.02)}")
    print(f"|t|>2: {sum(1 for x in t_vals if x > 2)}")

    rows = []
    for name, segs in results.items():
        row = {'factor': name}
        for seg_name in ['2015-2020', '2021-2026', 'full']:
            for h in horizons:
                key = f"{seg_name}_h{h}"
                s = segs.get(key, {})
                row[f'{key}_ic'] = s.get('ic_mean', 0)
                row[f'{key}_ir'] = s.get('ic_ir', 0)
                row[f'{key}_t'] = s.get('t_value', 0)
                row[f'{key}_win'] = s.get('win_rate', 0)
                row[f'{key}_n'] = s.get('n', 0)
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUTPUT / "factor_summary_3x3.csv", index=False)
    print(f"汇总已保存: {OUTPUT / 'factor_summary_3x3.csv'}", flush=True)


def fast_rank_ic_with_mask(factor, r_centered, r_norm, valid_day):
    """返回带NaN的IC数组（保留日期对齐，支持分段切分）"""
    T, N = factor.shape
    
    f_df = pd.DataFrame(factor)
    f_rank = f_df.rank(axis=1).values.astype(np.float64)
    
    mask = ~np.isnan(factor) & (r_norm > 0)[:, None]
    f_rank = np.where(mask, f_rank, 0.0)
    
    n_valid = mask.sum(axis=1)
    f_mean = f_rank.sum(axis=1) / np.maximum(n_valid, 1)
    f_centered = f_rank - f_mean[:, None]
    f_centered = np.where(mask, f_centered, 0.0)
    f_norm = np.sqrt((f_centered ** 2).sum(axis=1))
    
    num = (f_centered * r_centered).sum(axis=1)
    den = f_norm * r_norm
    ic = np.where(valid_day & (den > 0), num / np.where(den > 0, den, 1), np.nan)
    return ic


if __name__ == "__main__":
    main()
