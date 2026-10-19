"""全局配置：路径、参数、假设"""
from pathlib import Path

# === 路径 ===
PROJECT_ROOT = Path(r"D:\gtja191")
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUT = PROJECT_ROOT / "output"

# === DolphinDB ===
import os
DDB_HOST = os.environ.get("DDB_HOST", "10.65.0.75")
DDB_PORT = int(os.environ.get("DDB_PORT", "8902"))
DDB_USER = os.environ.get("DDB_USER", "read")
DDB_PASSWORD = os.environ.get("DDB_PASSWORD", "")
DDB_DATABASE = "dfs://d1_day"
DDB_STOCK_TABLE = "stock_daily"
DDB_INDEX_TABLE = "index_daily"

# === 数据参数 ===
DATA_START = "2014-01-01"
DATA_END = "2026-08-31"
BACKTEST_START = "2015-02-01"
BACKTEST_END = "2026-08-31"

BENCHMARK_INDEX = "000905.SH"  # 中证500
UNIVERSE = "all_a"  # 全A股

# === 函数语义约定 ===
STD_DDOF = 1
COV_DDOF = 1
CORR_METHOD = "pearson"
CORR_ZERO_VAR_RETURN = 0.0
WINDOW_INCLUDES_TODAY = True
TRADING_DAY_COUNT = True  # 窗口按交易日

SMA_INITIAL = "first_valid"  # SMA初值为首个有效值
SMA_SAME_DAY = True  # SMA同日输出
WMA_NORMALIZE = True  # WMA有限窗口归一化
DECAYLINEAR_LATEST_MAX = True  # 当日权重最大
REGBETA_INTERCEPT = True  # 含截距
REGBETA_RANK_INSUFFICIENT = 0.0  # 秩不足返回0

RANK_NORMALIZE = True  # 归一化到[0,1]
RANK_FROM_ZERO = True  # 从0开始
RANK_TIES = "average"  # 并列取平均

STOP_FILL_DAYS = 5  # 停牌前向填充最多5日

# === 分类C缺项默认参数 ===
DEFAULT_WINDOW = 20  # 缺失窗口的默认值
SMA_DEFAULT_N = 10
SMA_DEFAULT_M = 2

# === 回测参数 ===
PREDICTION_HORIZON = 2  # 预测周期（日）
TRANSACTION_COST = 0.003  # 单边0.3%
