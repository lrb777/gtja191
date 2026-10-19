# coding=utf-8
"""
Step3: 行业+市值中性化后的IC分析（修正版）

修正：IC计算改为逐日对齐排名，确保因子和收益在同一股票集合上排名
"""
import sys, os, time, warnings, json
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from config.settings import DATA_PROCESSED, DATA_RAW, OUTPUT

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


def load_industry_mapping(stocks):
    """加载申万一级行业分类"""
    raw_path = DATA_RAW / "申万一级行业分类_最新版.csv"
    ind_df = pd.read_csv(raw_path, encoding='utf-8')

    code_col = ind_df.columns[0]
    name_col = ind_df.columns[2]

    ind_df = ind_df[[code_col, name_col]].copy()
    ind_df.columns = ['ts_code', 'industry']
    ind_df = ind_df.drop_duplicates(subset='ts_code')

    ind_codes = pd.Categorical(ind_df['industry']).codes
    ind_map = dict(zip(ind_df['ts_code'], ind_codes))

    industry = np.full(len(stocks), -1, dtype=np.int32)
    for i, s in enumerate(stocks):
        if s in ind_map:
            industry[i] = ind_map[s]

    n_ind = max(ind_codes.max() + 1, 1)
    return industry, n_ind


def build_dummy_matrix(industry, n_ind):
    """构建行业哑变量矩阵 N×(n_ind-1)"""
    N = len(industry)
    dummies = np.zeros((N, n_ind - 1), dtype=np.float64)
    for k in range(1, n_ind):
        dummies[:, k - 1] = (industry == k).astype(np.float64)
    return dummies


def neutralize_factor(factor, dummies, log_mv):
    """逐日截面OLS回归取残差"""
    T, N = factor.shape
    residual = np.full_like(factor, np.nan)

    for t in range(T):
        f = factor[t]
        mv = log_mv[t]
        mask = ~(np.isnan(f) | np.isnan(mv))
        nv = mask.sum()
        if nv < 50:
            continue

        f_v = f[mask]
        mv_v = mv[mask]
        D_v = dummies[mask]

        X = np.column_stack([D_v, mv_v, np.ones(nv)])
        XtX = X.T @ X
        Xty = X.T @ f_v
        try:
            beta = np.linalg.solve(XtX, Xty)
            pred = X @ beta
            residual[t, mask] = f_v - pred
        except np.linalg.LinAlgError:
            continue

    return residual


def correct_rank_ic(factor, fwd_ret):
    """正确IC计算：每日截面在相同有效股票集合上同时排名"""
    T, N = factor.shape
    ic_list = []

    for t in range(T):
        f = factor[t]
        r = fwd_ret[t]
        mask = ~(np.isnan(f) | np.isnan(r))
        nv = mask.sum()
        if nv < 20:
            continue

        fv = f[mask]
        rv = r[mask]

        fr = fv.argsort().argsort().astype(np.float64)
        rr = rv.argsort().argsort().astype(np.float64)

        fr -= fr.mean()
        rr -= rr.mean()

        num = np.dot(fr, rr)
        den = np.sqrt(np.dot(fr, fr) * np.dot(rr, rr))

        if den > 0:
            ic_list.append(num / den)

    return np.array(ic_list)


def correct_rank_ic_segmented(factor, fwd_ret, seg_masks):
    """分段IC计算，返回每段IC统计"""
    results = {}
    for seg_name, seg_mask in seg_masks:
        seg_f = factor[seg_mask]
        seg_r = fwd_ret[seg_mask]
        ic_arr = correct_rank_ic(seg_f, seg_r)
        results[seg_name] = compute_ic_stats(ic_arr)
    return results


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
    print("=" * 60)
    print("Step3: 行业+市值中性化IC分析（修正版）")
    print("=" * 60)

    print("\n加载面板数据...", flush=True)
    panel = np.load(DATA_PROCESSED / "panel.npz")
    dates = np.load(DATA_PROCESSED / "dates.npy", allow_pickle=True)
    stocks = np.load(DATA_PROCESSED / "stocks.npy", allow_pickle=True)
    T, N = panel["close"].shape
    print(f"  T={T}, N={N}", flush=True)

    data = {k: panel[k] for k in panel.files}
    close = data["close"]
    circ_mv = data["circ_mv"]

    # 行业分类
    print("加载行业分类...", flush=True)
    industry, n_ind = load_industry_mapping(stocks)
    valid_ind = (industry >= 0).sum()
    print(f"  行业数: {n_ind}, 有行业分类的股票: {valid_ind}/{N}", flush=True)
    dummies = build_dummy_matrix(industry, n_ind)

    # log市值
    print("计算log市值...", flush=True)
    log_mv = np.log(circ_mv)
    log_mv = np.where(circ_mv > 0, log_mv, np.nan)

    # 时间段
    date_arr = pd.to_datetime(dates)
    seg1_mask = (date_arr < pd.Timestamp("2021-01-01"))
    seg2_mask = (date_arr >= pd.Timestamp("2021-01-01"))
    all_mask = np.ones(T, dtype=bool)
    seg_masks = [('2015-2020', seg1_mask), ('2021-2026', seg2_mask), ('full', all_mask)]

    # 前瞻收益
    horizons = [2, 5, 10]
    fwd_rets = {}
    for h in horizons:
        fr = np.full_like(close, np.nan)
        fr[:-h] = close[h:] / close[:-h] - 1
        fwd_rets[h] = fr

    # 加载原始IC用于对比
    raw_ic_path = OUTPUT / "ic_results_3x3.json"
    raw_ic = {}
    if os.path.exists(raw_ic_path):
        with open(raw_ic_path, 'r', encoding='utf-8') as f:
            raw_ic = json.load(f)
        print(f"  已加载原始IC: {len(raw_ic)} 个因子", flush=True)

    # 计算因子
    print(f"\n计算191因子 + 中性化 + IC（修正版）...", flush=True)
    all_funcs = get_all_alpha_funcs()
    print(f"  共 {len(all_funcs)} 个因子\n", flush=True)

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

            # 中性化
            t1 = time.time()
            factor_neut = neutralize_factor(factor, dummies, log_mv)
            dt_neut = time.time() - t1

            # IC计算（修正版：逐日对齐排名）
            results[name] = {}
            for h in horizons:
                fr = fwd_rets[h]
                for seg_name, seg_mask in seg_masks:
                    seg_f = factor_neut[seg_mask]
                    seg_r = fr[seg_mask]
                    ic_arr = correct_rank_ic(seg_f, seg_r)
                    key = f"{seg_name}_h{h}"
                    results[name][key] = compute_ic_stats(ic_arr)

            n_ok += 1
            dt_total = time.time() - t0
            if dt_total > 15:
                print(f"  SLOW: {name} total={dt_total:.1f}s (factor={dt_factor:.1f}s, neut={dt_neut:.1f}s, ic={dt_total-dt_neut-dt_factor:.1f}s)", flush=True)
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

    # 保存
    outpath = OUTPUT / "ic_results_neutralized.json"
    with open(outpath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"中性化IC已保存: {outpath}", flush=True)

    # 对比表
    print("\n" + "=" * 100)
    print("=== 中性化IC vs 原始IC 对比（全期_2日，按|中性化IC|降序前30）===")
    print("=" * 100)
    print(f"{'因子':>10s} | {'原始IC':>8s} {'中性IC':>8s} {'变化%':>7s} | {'原始t':>8s} {'中性t':>8s} | {'原始IR':>7s} {'中性IR':>7s}")
    print("-" * 100)

    sorted_factors = sorted(results.items(),
                           key=lambda x: abs(x[1].get('full_h2', {}).get('ic_mean', 0)),
                           reverse=True)
    for name, segs in sorted_factors[:30]:
        v = segs.get
        n_ic = v('full_h2', {}).get('ic_mean', 0)
        n_t = v('full_h2', {}).get('t_value', 0)
        n_ir = v('full_h2', {}).get('ic_ir', 0)

        raw_s = raw_ic.get(name, {}).get('full_h2', {})
        r_ic = raw_s.get('ic_mean', 0)
        r_t = raw_s.get('t_value', 0)
        r_ir = raw_s.get('ic_ir', 0)

        change = ((n_ic - r_ic) / abs(r_ic) * 100) if abs(r_ic) > 0.001 else 0
        print(f"{name:>10s} | {r_ic:>+8.4f} {n_ic:>+8.4f} {change:>+6.1f}% | {r_t:>+8.1f} {n_t:>+8.1f} | {r_ir:>+7.3f} {n_ir:>+7.3f}")

    # 总体统计
    print(f"\n=== 总体统计（全期_2日）===")
    raw_ics = [abs(raw_ic.get(n, {}).get('full_h2', {}).get('ic_mean', 0)) for n in results]
    neut_ics = [abs(v.get('full_h2', {}).get('ic_mean', 0)) for v in results.values()]
    raw_ts = [abs(raw_ic.get(n, {}).get('full_h2', {}).get('t_value', 0)) for n in results]
    neut_ts = [abs(v.get('full_h2', {}).get('t_value', 0)) for v in results.values()]

    print(f"{'指标':>15s} | {'原始':>8s} {'中性化':>8s}")
    print("-" * 40)
    print(f"{'|IC|均值':>15s} | {np.mean(raw_ics):>8.4f} {np.mean(neut_ics):>8.4f}")
    print(f"{'|IC|>0.02':>15s} | {sum(1 for x in raw_ics if x > 0.02):>8d} {sum(1 for x in neut_ics if x > 0.02):>8d}")
    print(f"{'|IC|>0.05':>15s} | {sum(1 for x in raw_ics if x > 0.05):>8d} {sum(1 for x in neut_ics if x > 0.05):>8d}")
    print(f"{'|t|>2':>15s} | {sum(1 for x in raw_ts if x > 2):>8d} {sum(1 for x in neut_ts if x > 2):>8d}")

    # CSV
    rows = []
    for name, segs in results.items():
        row = {'factor': name}
        for seg_name in ['2015-2020', '2021-2026', 'full']:
            for h in horizons:
                key = f"{seg_name}_h{h}"
                s = segs.get(key, {})
                raw_s = raw_ic.get(name, {}).get(key, {})
                row[f'{key}_neut_ic'] = s.get('ic_mean', 0)
                row[f'{key}_raw_ic'] = raw_s.get('ic_mean', 0)
                row[f'{key}_neut_t'] = s.get('t_value', 0)
                row[f'{key}_raw_t'] = raw_s.get('t_value', 0)
                row[f'{key}_neut_ir'] = s.get('ic_ir', 0)
                row[f'{key}_raw_ir'] = raw_s.get('ic_ir', 0)
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUTPUT / "factor_summary_neutralized.csv", index=False)
    print(f"\n对比汇总已保存: {OUTPUT / 'factor_summary_neutralized.csv'}", flush=True)


if __name__ == "__main__":
    main()
