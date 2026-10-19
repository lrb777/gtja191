# coding=utf-8
"""
Step0: 从 DolphinDB 获取 Alpha191 所需数据（2014-2026）
分年拉取全A个股 + 中证500指数
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from src.data.ddb_client import DDBClient
from config.settings import BENCHMARK_INDEX, DATA_RAW

YEARS = list(range(2014, 2027))


def main():
    with DDBClient() as c:
        all_stocks = []
        for year in YEARS:
            start = f"{year}-01-01"
            end = f"{year}-12-31" if year < 2026 else "2026-08-31"
            print(f"获取 {start} ~ {end} ...", end=" ", flush=True)
            df = c.get_stock_daily(start, end)
            print(f"{len(df)} 条, {df['ts_code'].nunique()} 只股票")
            all_stocks.append(df)

        stock_df = pd.concat(all_stocks, ignore_index=True)
        print(f"\n合计: {len(stock_df)} 条, {stock_df['ts_code'].nunique()} 只股票")
        print(f"日期范围: {stock_df['trade_date'].min().date()} ~ {stock_df['trade_date'].max().date()}")

        out_path = DATA_RAW / "stock_daily.parquet"
        stock_df.to_parquet(out_path, index=False)
        print(f"已保存: {out_path} ({os.path.getsize(out_path) / 1024 / 1024:.1f} MB)")

        print(f"\n获取中证500指数日线 ...")
        index_df = c.get_index_daily(BENCHMARK_INDEX, "2014-01-01", "2026-08-31")
        print(f"  {len(index_df)} 条")
        out_path = DATA_RAW / "index_500.parquet"
        index_df.to_parquet(out_path, index=False)
        print(f"  已保存: {out_path}")

        trade_dates = c.get_trade_dates("2014-01-01", "2026-08-31")
        print(f"\n交易日: {len(trade_dates)} 天")
        pd.DataFrame({"trade_date": trade_dates}).to_csv(DATA_RAW / "trade_dates.csv", index=False)

        print(f"\n=== 数据概览 ===")
        print(f"股票数: {stock_df['ts_code'].nunique()}")
        print(f"交易日数: {len(trade_dates)}")
        print(f"总记录数: {len(stock_df)}")


if __name__ == "__main__":
    main()
