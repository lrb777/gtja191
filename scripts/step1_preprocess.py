# coding=utf-8
"""
Step1: 数据预处理（向量化版）
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from config.settings import DATA_RAW, DATA_PROCESSED, BACKTEST_START, STOP_FILL_DAYS


def main():
    print("加载原始数据...")
    df = pd.read_parquet(DATA_RAW / "stock_daily.parquet")
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    print(f"  {len(df)} 条, {df['ts_code'].nunique()} 只股票")

    print("加载流通市值数据...")
    circ = pd.read_parquet(DATA_RAW / "circ_mv.parquet")
    circ["trade_date"] = pd.to_datetime(circ["trade_date"])
    df = df.merge(circ, on=["ts_code", "trade_date"], how="left")
    print(f"  circ_mv: {df['circ_mv'].notna().sum()} / {len(df)} 非 NaN")

    # === 1. 前复权 ===
    print("\n前复权处理...")
    df["adj"] = df["adj_factor"].astype(float)
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float) * df["adj"]
    df["vol"] = df["vol"].astype(float) / df["adj"]
    df["amount"] = df["amount"].astype(float) / df["adj"]

    # === 2. VWAP ===
    print("计算 VWAP...")
    df["vwap"] = df["amount"] / df["vol"]
    df.loc[df["vol"] == 0, "vwap"] = np.nan

    # === 3. 股票池过滤 ===
    print("股票池过滤...")
    df = df[df["vol"] > 0].copy()
    df["is_limit_up"] = (df["close"] >= df["up_limit"].astype(float) * 0.999) & (df["close"] > df["open"])
    df["is_limit_dn"] = (df["close"] <= df["down_limit"].astype(float) * 1.001) & (df["close"] < df["open"])
    df = df[~(df["is_limit_up"] | df["is_limit_dn"])].copy()
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    df["stock_day"] = df.groupby("ts_code").cumcount()
    df = df[df["stock_day"] >= 60].copy()
    bt_start = pd.Timestamp(BACKTEST_START)
    df = df[df["trade_date"] >= bt_start].copy()
    print(f"  过滤后: {len(df)} 条, {df['ts_code'].nunique()} 只股票")

    # === 4. 构建面板（pivot向量化）===
    print("\n构建面板数据...")
    dates = sorted(df["trade_date"].unique())
    stocks = sorted(df["ts_code"].unique())
    T, N = len(dates), len(stocks)
    print(f"  T={T} 交易日, N={N} 股票")

    date_idx = {d: i for i, d in enumerate(dates)}
    stock_idx = {s: i for i, s in enumerate(stocks)}

    # 用 categorical 编码加速
    df["date_i"] = df["trade_date"].map(date_idx)
    df["stock_i"] = df["ts_code"].map(stock_idx)

    fields = ["open", "high", "low", "close", "vol", "amount", "vwap", "total_mv", "circ_mv"]
    panel = {}
    for f in fields:
        arr = np.full((T, N), np.nan, dtype=np.float64)
        arr[df["date_i"].values, df["stock_i"].values] = df[f].values
        panel[f] = arr

    # === 5. 前向填充 ===
    print(f"前向填充（最多{STOP_FILL_DAYS}日）...")
    for f in fields:
        arr = panel[f]
        for t in range(1, T):
            mask = np.isnan(arr[t])
            if mask.any():
                arr[t, mask] = arr[t - 1, mask]
        panel[f] = arr

    # === 6. 收益率 ===
    print("计算收益率...")
    ret = np.full((T, N), np.nan, dtype=np.float64)
    ret[1:] = panel["close"][1:] / panel["close"][:-1] - 1
    panel["ret"] = ret

    # === 7. 基准指数 ===
    print("加载中证500指数...")
    idx_df = pd.read_parquet(DATA_RAW / "index_500.parquet")
    idx_df["trade_date"] = pd.to_datetime(idx_df["trade_date"])
    idx_df = idx_df[idx_df["trade_date"].isin(dates)].sort_values("trade_date").reset_index(drop=True)
    print(f"  {len(idx_df)} 条")

    bm_open = np.full((T, N), np.nan, dtype=np.float64)
    bm_close = np.full((T, N), np.nan, dtype=np.float64)
    for _, row in idx_df.iterrows():
        i = date_idx.get(row["trade_date"])
        if i is not None:
            bm_open[i, :] = row["open"]
            bm_close[i, :] = row["close"]
    for t in range(1, T):
        mask = np.isnan(bm_open[t])
        bm_open[t, mask] = bm_open[t-1, mask]
        mask = np.isnan(bm_close[t])
        bm_close[t, mask] = bm_close[t-1, mask]
    panel["benchmark_open"] = bm_open
    panel["benchmark_close"] = bm_close

    # === 8. 保存 ===
    outpath = DATA_PROCESSED / "panel.npz"
    np.savez_compressed(outpath, **panel)
    print(f"\n已保存: {outpath} ({os.path.getsize(outpath) / 1024 / 1024:.1f} MB)")
    np.save(DATA_PROCESSED / "dates.npy", np.array(dates))
    np.save(DATA_PROCESSED / "stocks.npy", np.array(stocks))
    print(f"已保存: dates.npy ({T}), stocks.npy ({N})")

    print(f"\n=== 面板概览 ===")
    for f in fields + ["ret"]:
        arr = panel[f]
        print(f"  {f}: NaN={np.isnan(arr).mean():.1%}, mean={np.nanmean(arr):.4f}")

if __name__ == "__main__":
    main()
