# GTJA Alpha191 因子复现

复现国泰君安《基于短周期价量特征的多因子选股体系》（2017年6月）中的191个Alpha因子。

## 项目概况

- **数据范围**：2014-01-01 ~ 2026-08-31，全A股（5765只）
- **数据源**：DolphinDB（`dfs://d1_day`），含OHLCV、复权因子、市值、涨跌停等
- **基准指数**：中证500（000905.SH）
- **因子实现**：191个Alpha因子，numpy向量化，基于 `sliding_window_view`
- **IC分析**：Rank IC（Spearman），3时间段 × 3换仓频率 = 9组
- **中性化**：行业（申万一级）+ 流通市值 OLS 回归取残差

## 目录结构

```
gtja191/
├── config/settings.py              # 全局配置（路径、DolphinDB连接、参数）
├── data/
│   ├── raw/                        # 原始数据（parquet/csv，不入git）
│   └── processed/                  # 预处理后面板（npz，不入git）
├── src/
│   ├── data/ddb_client.py          # DolphinDB数据访问层
│   ├── lib/
│   │   ├── ts_operators.py         # 时序算子（DELAY/DELTA/SUM/CORR/STD/TSRANK/SMA等）
│   │   ├── cross_section.py        # 截面算子（RANK/REGBETA/REGRESI/SEQUENCE）
│   │   └── special.py              # 特殊函数（DTM/DBM/SELF）
│   └── factors/
│       ├── alpha_001_050.py        # Alpha 1-50
│       ├── alpha_051_100.py        # Alpha 51-100
│       ├── alpha_101_150.py        # Alpha 101-150
│       └── alpha_151_191.py        # Alpha 151-191
├── scripts/
│   ├── step0_fetch_data.py         # 数据拉取（DolphinDB → parquet）
│   ├── step1_preprocess.py         # 预处理（复权、VWAP、过滤、面板构建）
│   ├── step2_compute_and_evaluate.py  # 因子计算 + 9组IC分析
│   └── step3_neutralize_ic.py      # 行业+市值中性化IC分析
├── tests/test_base_functions.py    # 单元测试（33项，全部通过）
├── output/                         # 结果输出（不入git）
├── gtja_alpha191_formula_audit.md  # 191因子公式审计
└── gtja_alpha191_implementation_plan.md  # 实现计划
```

## 执行流程

```bash
# 1. 拉取数据（需DolphinDB连接）
python scripts/step0_fetch_data.py

# 2. 预处理（复权、VWAP、股票池过滤、面板构建）
python scripts/step1_preprocess.py

# 3. 因子计算 + IC分析（3段×3频率）
python scripts/step2_compute_and_evaluate.py

# 4. 行业+市值中性化IC分析
python scripts/step3_neutralize_ic.py
```

## IC分析设计

### 时间段
| 段 | 区间 | 说明 |
|---|---|---|
| 2015-2020 | 2015.02 ~ 2020.12 | 历史样本 |
| 2021-2026 | 2021.01 ~ 2026.08 | 近年样本 |
| full | 2015.02 ~ 2026.08 | 全期 |

### 换仓频率
- h=2：2日前瞻收益
- h=5：5日前瞻收益
- h=10：10日前瞻收益

### IC统计量
- IC均值、IC标准差、ICIR、t值、胜率、样本数

### 中性化
- **行业**：申万一级行业（31类），哑变量回归
- **市值**：log(流通市值)
- **方法**：每日截面 OLS 回归，`factor ~ industry_dummies + log(circ_mv) + 1`，取残差

## 环境配置

```bash
# DolphinDB连接（环境变量）
set DDB_PASSWORD=your_password

# Python依赖
pip install numpy pandas dolphindb
```

## 文档

- `gtja_alpha191_formula_audit.md` — 191因子公式完备性审计（A/B/C/D分类）
- `gtja_alpha191_implementation_plan.md` — 实现计划与函数语义说明
