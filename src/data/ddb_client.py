"""DolphinDB 数据访问层（Alpha191专用）。

封装 dfs://d1_day 的连接与查询，提供：
- get_stock_daily:   全A个股日频OHLCV+复权因子+涨跌停
- get_index_daily:   指数日线（中证500 OHLCV）
- get_trade_dates:   交易日列表
- get_stock_list:    全A股代码列表
"""
import dolphindb as ddb
import pandas as pd
from typing import Optional
from config.settings import (
    DDB_HOST, DDB_PORT, DDB_USER, DDB_PASSWORD,
    DDB_DATABASE, DDB_STOCK_TABLE, DDB_INDEX_TABLE
)


class DDBClient:
    def __init__(self, host: str = DDB_HOST, port: int = DDB_PORT,
                 user: str = DDB_USER, password: str = DDB_PASSWORD,
                 database: str = DDB_DATABASE):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self._session: Optional[ddb.session] = None

    @property
    def session(self) -> ddb.session:
        if self._session is None:
            self._session = ddb.session()
            self._session.connect(self.host, self.port, self.user, self.password)
        return self._session

    def close(self):
        if self._session is not None:
            self._session.close()
            self._session = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _fmt_date(self, d: str) -> str:
        d = d.replace("-", "")
        return f"{d[:4]}.{d[4:6]}.{d[6:]}"

    def get_stock_daily(self, start: str, end: str,
                        ts_codes: Optional[list] = None) -> pd.DataFrame:
        """获取全A个股日频数据（原始未复权）。

        Returns:
            DataFrame[ts_code, trade_date, open, high, low, close,
                       vol, amount, adj_factor, turnover_rate,
                       up_limit, down_limit, total_mv]
        """
        sd = self._fmt_date(start)
        ed = self._fmt_date(end)

        code_filter = ""
        if ts_codes is not None and len(ts_codes) > 0:
            codes_str = ", ".join([f'"{c}"' for c in ts_codes])
            code_filter = f"and ts_code in [{codes_str}]"

        df = self.session.run(f'''
            select ts_code, trade_date, open, high, low, close,
                   vol, amount, adj_factor, turnover_rate,
                   up_limit, down_limit, total_mv, circ_mv
            from loadTable("{self.database}", "{DDB_STOCK_TABLE}")
            where trade_date >= {sd}
              and trade_date <= {ed}
              {code_filter}
            order by ts_code, trade_date
        ''')
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        return df

    def get_index_daily(self, ts_code: str, start: str, end: str) -> pd.DataFrame:
        """获取指数日线。

        Returns:
            DataFrame[trade_date, open, high, low, close, vol, amount]
        """
        sd = self._fmt_date(start)
        ed = self._fmt_date(end)
        df = self.session.run(f'''
            select trade_date, open, high, low, close, vol, amount
            from loadTable("{self.database}", "{DDB_INDEX_TABLE}")
            where ts_code = "{ts_code}"
              and trade_date >= {sd}
              and trade_date <= {ed}
            order by trade_date
        ''')
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        return df

    def get_trade_dates(self, start: str, end: str) -> list:
        """获取交易日列表。"""
        sd = self._fmt_date(start)
        ed = self._fmt_date(end)
        df = self.session.run(f'''
            select distinct trade_date from loadTable("{self.database}", "{DDB_INDEX_TABLE}")
            where trade_date >= {sd} and trade_date <= {ed}
            order by trade_date
        ''')
        return pd.to_datetime(df["trade_date"]).tolist()

    def get_stock_list(self, date: str) -> list:
        """获取某日有数据的全部A股代码。"""
        d = self._fmt_date(date)
        df = self.session.run(f'''
            select distinct ts_code from loadTable("{self.database}", "{DDB_STOCK_TABLE}")
            where trade_date = {d}
            order by ts_code
        ''')
        return df["ts_code"].tolist()
