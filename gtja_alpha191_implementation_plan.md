# GTJA Alpha191 因子复现实现计划框架

> **版本**: v1.0  
> **日期**: 2026-10-12  
> **目标**: 复现国泰君安《基于短周期价量特征的多因子选股体系》（2017年6月）中的191个Alpha因子  
> **基准**: 中证500 | **预测周期**: 2日 | **交易成本**: Tc=0.3%

---

## 目录

1. [总体架构设计](#1-总体架构设计)
2. [阶段0：基础设施与数据准备](#2-阶段0基础设施与数据准备)
3. [阶段1：函数库实现](#3-阶段1函数库实现)
4. [阶段2：因子分层实现](#4-阶段2因子分层实现)
5. [阶段3：因子检验](#5-阶段3因子检验)
6. [阶段4：策略复现](#6-阶段4策略复现)
7. [分类B：语义歧义——推荐选择与理由](#7-分类b语义歧义推荐选择与理由)
8. [分类C：22条因子专属缺项——候选修正方案](#8-分类c22条因子专属缺项候选修正方案)
9. [分类D：代数冗余/尺度敏感——处理方案](#9-分类d代数冗余尺度敏感处理方案)
10. [风险与假设清单](#10-风险与假设清单)
11. [验证与对齐检查点](#11-验证与对齐检查点)
12. [附录](#12-附录)

---

## 1. 总体架构设计

### 1.1 代码模块划分

```
gtja_alpha191/
├── config/
│   ├── settings.py              # 全局配置（路径、参数、假设）
│   └── factor_registry.py       # 191因子注册表（分类、依赖、参数）
├── data/
│   ├── fetcher.py               # 数据获取（Wind/本地CSV）
│   ├── preprocessor.py          # 复权、清洗、对齐
│   ├── universe.py              # 股票池构建（去ST/停牌/涨跌停）
│   └── benchmark.py             # 基准指数 & Fama-French三因子
├── lib/
│   ├── base_functions.py        # 附录30个基础函数实现
│   ├── ts_operators.py          # 时序算子（DELTA, DELAY, TSMAX, TSMIN等）
│   ├── cross_section.py         # 截面算子（RANK, REGBETA, REGRESI等）
│   ├── rolling.py               # 滚动窗口工具
│   └── validation.py            # 函数单元测试与边界验证
├── factors/
│   ├── alpha_formulas.py        # 191因子公式定义（声明式DSL）
│   ├── category_a.py            # 分类A：140条（完整，需统一口径）
│   ├── category_b.py            # 分类B：20条（语义歧义修正版）
│   ├── category_c.py            # 分类C：22条（缺项修正版）
│   ├── category_d.py            # 分类D：9条（冗余简化版）
│   └── compute.py               # 因子计算引擎（批量、增量）
├── analysis/
│   ├── ic_analysis.py           # IC/IR分析
│   ├── group_return.py          # 分组收益
│   ├── factor_corr.py           # 因子相关性矩阵
│   ├── significance.py          # 显著性检验
│   └── turnover.py              # 换手率分析
├── strategy/
│   ├── neutralization.py        # 行业中性化 + BARRA CNE5风格中性化
│   ├── factor_synthesis.py      # 因子合成（等权/IC加权/最优化）
│   ├── portfolio.py             # 组合优化（约束优化）
│   └── backtest.py              # 回测引擎
├── tests/
│   ├── test_base_functions.py   # 基础函数单元测试
│   ├── test_factors.py          # 因子输出验证
│   └── test_replication.py      # 研报复现对齐测试
└── outputs/
    ├── figures/                 # 图表输出
    └── reports/                 # 报告输出
```

### 1.2 数据流设计

```
[Wind/本地数据源]
        │
        ▼
┌──────────────────┐
│  data/fetcher.py │  ← 日频OHLCV + 基准指数 + 三因子
└──────┬───────────┘
       │
       ▼
┌──────────────────────┐
│ data/preprocessor.py │  ← 前复权、缺失处理、对齐
└──────┬───────────────┘
       │
       ▼
┌───────────────────┐
│ data/universe.py  │  ← 股票池过滤（非ST、非停牌、非涨跌停）
└──────┬────────────┘
       │
       ▼
┌─────────────────────────┐
│ lib/base_functions.py   │  ← 30个基础函数（向量化实现）
└──────┬──────────────────┘
       │
       ▼
┌─────────────────────────┐
│ factors/compute.py      │  ← 批量计算191因子（按分类分层）
└──────┬──────────────────┘
       │
       ▼
┌─────────────────────────┐
│ analysis/*.py           │  ← IC分析、分组收益、相关性
└──────┬──────────────────┘
       │
       ▼
┌─────────────────────────┐
│ strategy/*.py           │  ← 中性化 → 合成 → 优化 → 回测
└──────┬──────────────────┘
       │
       ▼
[回测结果：年化超额、最大回撤、IR、换手率]
```

### 1.3 技术栈建议

| 层级 | 技术选择 | 理由 |
|------|---------|------|
| 语言 | Python 3.10+ | 生态成熟，pandas/numpy向量化性能足够 |
| 数据处理 | pandas 2.x + numpy | 面板数据操作标准工具 |
| 数值计算 | numba (JIT加速滚动窗口) | 191因子大量滚动计算，numba可显著提速 |
| 因子存储 | HDF5 (pandas HDFStore) 或 Parquet | 支持多维索引、增量写入、快速切片 |
| 数据源 | DolphinDB (`dfs://d1_day`) | 日频OHLCV+复权因子+涨跌停+换手率+市值 |
| 组合优化 | cvxpy 或 scipy.optimize | 约束优化求解 |
| 回测框架 | 自建轻量引擎 | 避免alphalens/zipline等框架的假设冲突 |
| 可视化 | matplotlib + seaborn | 标准科研绘图 |
| 测试 | pytest | 单元测试与回归测试 |
| 配置管理 | YAML + dataclasses | 参数集中管理、类型安全 |

### 1.4 关键设计原则

1. **声明式因子定义**：每个因子用公式字符串 + 参数字典描述，由计算引擎统一解析执行，避免逐因子手写代码。
2. **向量化优先**：所有基础函数必须支持面板向量化（`(T, N)` 形状输入），仅在必要时使用 `groupby-rolling`。
3. **可复现性**：所有随机种子固定（`np.random.seed(42)`），所有假设显式记录在 `config/settings.py`。
4. **增量计算**：支持按日期增量追加因子值，避免每次全量重算。
5. **分层验证**：每层因子计算后立即运行验证脚本，确保输出形状、数值范围、NA比例合理。

---

## 2. 阶段0：基础设施与数据准备

### 2.1 数据获取

#### 2.1.1 日频OHLCV数据

| 字段 | 说明 | 来源 |
|------|------|------|
| OPEN | 开盘价 | DolphinDB `stock_daily.open` × `adj_factor`（前复权） |
| HIGH | 最高价 | 同上 |
| LOW | 最低价 | 同上 |
| CLOSE | 收盘价 | 同上 |
| VOLUME | 成交量（股） | DolphinDB `stock_daily.vol` / `adj_factor` |
| AMOUNT | 成交额（元） | DolphinDB `stock_daily.amount` / `adj_factor` |
| VWAP | 均价 | 计算：`AMOUNT / VOLUME`（见假设2.3.2） |
| TURN | 换手率 | DolphinDB `stock_daily.turnover_rate` |
| TRADING_STATUS | 交易状态 | vol=0 或 hq_* 为 NaN 判定停牌 |
| ST_STATUS | ST标识 | 股票名称变更跟踪 |
| ADJ_FACTOR | 复权因子 | DolphinDB `stock_daily.adj_factor` |
| UP_LIMIT | 涨停价 | DolphinDB `stock_daily.up_limit` |
| DOWN_LIMIT | 跌停价 | DolphinDB `stock_daily.down_limit` |

**时间范围**：
- 因子计算：2010-01-01 至 2017-04-30（含预热期约500个交易日）
- 回测期：2012-01-01 至 2017-04-30（与研报对齐）

**股票范围**：全A股（沪深主板 + 创业板 + 科创板前身）

#### 2.1.2 基准指数数据

| 字段 | 说明 |
|------|------|
| 000905.SH (中证500) | CLOSE序列，用于基准比较和Alpha75/149/181/182 |
| 000905.SH OPEN | 用于BANCHMARKINDEXOPEN（注意研报命名反了，见分类B） |

#### 2.1.3 Fama-French三因子（Alpha30需要）

| 因子 | 说明 | 构建方式 |
|------|------|---------|
| MKT | 市场超额收益 | 全A等权收益 - 无风险利率 |
| SMB | 规模因子 | 小市值组合 - 大市值组合（市值中位数分组） |
| HML | 价值因子 | 高BM组合 - 低BM组合（BM 30%/70%分位分组） |

**构建细节**（研报未披露，需假设）：
- 股票域：全A股（与因子股票池一致）
- 分组方式：市值按中位数、BM按30%/70%分位
- 组合加权：等权
- 无风险利率：一年期定存利率（日化）
- 再平衡频率：月度

### 2.2 复权处理

**假设（置信度：高）**：采用**前复权**。

理由：
1. 前复权保持当前价格为真实交易价格，适合模拟实际交易
2. 国内量化研究惯例以前复权为主
3. 研报中 `OPEN/CLOSE(-1)` 等涉及跨日价格比较的因子，前复权可避免除权缺口

**实现**：
```python
# DolphinDB 已有 adj_factor 字段，前复权直接乘
df['open'] = df['open'] * df['adj_factor']
df['high'] = df['high'] * df['adj_factor']
df['low'] = df['low'] * df['adj_factor']
df['close'] = df['close'] * df['adj_factor']
df['vwap'] = df['amount'] / df['vol']  # 复权后重算
df['vol'] = df['vol'] / df['adj_factor']
df['amount'] = df['amount'] / df['adj_factor']
```

### 2.3 股票池构建

#### 2.3.1 过滤规则

| 规则 | 实现 |
|------|------|
| 剔除ST/*ST | `ST_STATUS == 1` 时剔除 |
| 剔除停牌 | `TRADING_STATUS == 0` 或 VOLUME==0 时剔除 |
| 剔除涨跌停 | 当日 `CLOSE == HIGH == LOW` 且涨跌幅绝对值 ≥ 9.5%（主板）/ 19.5%（科创创业） |
| 剔除上市初期 | 上市后前60个交易日剔除（避免新股效应） |

#### 2.3.2 VWAP定义

**假设（置信度：高）**：`VWAP = AMOUNT / VOLUME`（日成交额 / 日成交量）。

理由：
1. 研报附录未定义VWAP函数，但因子中直接使用VWAP变量
2. 日频VWAP最自然的定义就是日均成交价
3. 若使用供应商VWAP字段（如Wind的VWAP），与AMOUNT/VOLUME可能存在微小差异（四舍五入），但影响可忽略

### 2.4 预热长度计算

每个因子的预热长度需沿表达式依赖链递归计算：

```
preheat(factor) = max(
    max(preheat(child) for child in dependencies),
    max(window_size for window in windows)
)
```

**关键因子的预热估计**：
- 大部分因子：250个交易日（MAX窗口）
- Alpha30（三因子回归）：500个交易日（回归稳定性）
- Alpha75/149/181/182（基准指数相关）：250个交易日

**总预热期**：500个交易日（约2年），即数据从2010-01-01开始。

---

## 3. 阶段1：函数库实现

### 3.1 函数清单与参数约定

以下按附录顺序列出所有函数，明确参数约定和边界处理。

#### 3.1.1 时序基础算子

| 函数 | 公式 | 参数约定 | 边界处理 |
|------|------|---------|---------|
| `DELTA(S, d)` | S(t) - S(t-d) | d≥1整数 | d=0返回0；不足d期返回NaN |
| `DELAY(S, d)` | S(t-d) | d≥0整数 | d=0返回S(t)；不足d期返回NaN |
| `SUM(S, n)` | ΣS(t-i), i=0..n-1 | n≥1整数 | 不足n期返回NaN |
| `MEAN(S, n)` | SUM(S,n)/n | n≥1整数 | 不足n期返回NaN |
| `STD(S, n)` | 样本标准差 | **ddof=1**（样本标准差） | n<2返回NaN；零方差返回0（非NaN） |
| `COVIANCE(S1, S2, n)` | 样本协方差 | **ddof=1** | n<2返回NaN |
| `CORR(S1, S2, n)` | Pearson相关系数 | **Pearson** | 任一序列零方差返回0（非NaN） |
| `MAX(S, n)` | 滚动窗口最大值 | n≥1整数 | 不足n期返回NaN |
| `MIN(S, n)` | 滚动窗口最小值 | n≥1整数 | 不足n期返回NaN |
| `TSMAX(S, n)` | 同MAX(S,n) | 同上 | 同上 |
| `TSMIN(S, n)` | 同MIN(S,n) | 同上 | 同上 |
| `TSRANK(S, n)` | 时序排名（百分位） | **归一化到[0,1]**，从0开始 | n<2返回NaN；并列取平均排名 |
| `SMA(S, n, m)` | S(t)=(m/n)*S_raw+(1-m/n)*S(t-1) | n>m>0 | 初值=S_raw(0)；同日输出（非滞后） |
| `WMA(S, n)` | 加权移动平均，权重n,n-1,...,1 | **有限窗口**，权重归一化 | 不足n期返回NaN |
| `DECAYLINEAR(S, n)` | 线性衰减加权，权重n,n-1,...,1 | **当日权重=n**（最近日权重最大） | 不足n期返回NaN |
| `SUMAC(S, n)` | 累计和 | **返回累计路径**（长度为n的序列） | 不足n期返回NaN |

#### 3.1.2 截面算子

| 函数 | 公式 | 参数约定 | 边界处理 |
|------|------|---------|---------|
| `RANK(S)` | 截面排名 | **归一化到[0,1]**，从0开始，并列取平均 | 全NaN返回NaN |
| `REGBETA(S, X, n)` | OLS回归系数β | **含截距**，等权 | 秩不足返回0；n<3返回NaN |
| `REGRESI(S, X, n)` | OLS残差 | **含截距**，等权 | 秩不足返回NaN；n<3返回NaN |
| `FILTER(S, n)` | 条件过滤 | 见3.1.4 | — |

#### 3.1.3 专用函数

| 函数 | 定义 | 参数约定 | 边界处理 |
|------|------|---------|---------|
| `HIGHDAY(S, n)` | S在n期内最大值距今的交易日数 | 返回整数（0表示当日） | 并列取最近一次 |
| `LOWDAY(S, n)` | S在n期内最小值距今的交易日数 | 返回整数（0表示当日） | 并列取最近一次 |
| `COUNT(S, n)` | S在n期内为True的次数 | n≥1整数 | 不足n期返回NaN |
| `SUMIF(S, cond, n)` | 满足条件时求和 | n≥1整数 | 不足n期返回NaN |
| `PROD(S, n)` | 连乘 | n≥1整数 | 不足n期返回NaN |
| `DTM(OPEN, n)` | 开盘价大于昨收的最大值 | 见3.1.5 | — |
| `DBM(OPEN, n)` | 开盘价小于昨收的最小值 | 见3.1.5 | — |
| `SELF(S, n)` | S的第n个值 | n≥0整数 | 超出范围返回NaN |

#### 3.1.4 FILTER函数

```python
def FILTER(S, n):
    """
    条件过滤：当S(t)为True时，未来n-1期强制为False。
    用于避免信号重复触发。
    """
    result = S.copy()
    for t in range(len(S)):
        if S[t]:
            result[t+1:t+n] = False
    return result
```

#### 3.1.5 DTM/DBM函数

```python
def DTM(OPEN, n):
    """
    上涨开盘缺口：max(Sj - Sj-1, 0) 其中 Sj = max(OPEN(t), CLOSE(t-1))
    当 OPEN(t) <= CLOSE(t-1) 时返回0
    """
    S = np.maximum(OPEN, shift(CLOSE, 1))
    delta = S - shift(S, 1)
    return np.where(OPEN > shift(CLOSE, 1), np.maximum(delta, 0), 0)

def DBM(OPEN, n):
    """
    下跌开盘缺口：当 OPEN(t) < CLOSE(t-1) 时，返回 OPEN(t) - CLOSE(t-1)（负值）
    注意：研报原文DBM定义为"开盘价小于昨收的最小值"，但实际应为开盘价与昨收的差值
    """
    return np.where(OPEN < shift(CLOSE, 1), OPEN - shift(CLOSE, 1), 0)
```

### 3.2 共同口径约定（分类A依赖）

| 约定项 | 选择 | 理由 |
|--------|------|------|
| STD ddof | **1**（样本标准差） | 金融时间序列标准做法 |
| COVIANCE ddof | **1** | 与STD一致 |
| CORR类型 | **Pearson** | 研报未提Spearman/Kendall，默认Pearson |
| 零方差CORR | **返回0** | 避免NaN传播，0表示无线性关系 |
| 窗口包含当日 | **是** | `SUM(S,5)` = S(t)+S(t-1)+...+S(t-4)，共5个值 |
| 交易日vs自然日 | **交易日** | 所有窗口按交易日计数 |
| 停牌处理 | **前向填充最多5日**，超过则NaN | 避免长停牌污染因子 |
| SMA初值 | **S_raw(0)**（第一个有效值） | 标准递推初值 |
| SMA输出 | **同日**（非滞后） | 递推式本身不含滞后 |
| WMA归一化 | **有限窗口内归一化** | 权重和=1 |
| DECAYLINEAR时间方向 | **当日权重=n**（最近日最大） | 符合"线性衰减"语义 |
| REGBETA/REGRESI截距 | **含截距** | 标准OLS |
| REGBETA/REGRESI权重 | **等权** | 研报未提WLS |
| REGBETA秩不足 | **返回0** | 避免NaN |
| RANK范围 | **[0, 1]**，从0开始，并列取平均 | 与TSRANK一致 |
| SUMAC | **返回累计路径**（序列） | 函数名暗示累计 |
| 条件表达式优先级 | **C语言优先级**（`&&` > `||`，`<` > `&&`） | 标准编程惯例 |
| 缺失条件输出 | **NaN** | 显式缺失 |

### 3.3 函数实现验证

每个函数实现后需通过以下测试：

```python
# 示例：STD函数测试
def test_std():
    # 1. 基本功能
    s = np.array([1, 2, 3, 4, 5])
    assert np.isclose(STD(s, 5), np.std(s, ddof=1))
    
    # 2. 不足窗口
    assert np.isnan(STD(s[:2], 5))
    
    # 3. 零方差
    assert STD(np.ones(5), 5) == 0.0
    
    # 4. 面板输入 (T, N)
    panel = np.random.randn(100, 500)
    result = STD(panel, 20)
    assert result.shape == (100, 500)
    assert not np.any(np.isinf(result))
```

---

## 4. 阶段2：因子分层实现

### 4.1 实现顺序与理由

```
分类A (140条) → 分类D (9条) → 分类B (20条) → 分类C (22条)
```

**理由**：
- **分类A**：公式完整，先实现可快速建立因子计算基础设施和验证框架
- **分类D**：代数冗余，在A的基础上做简化，改动最小
- **分类B**：语义歧义，需要逐因子确认MAX/MIN含义，但公式结构完整
- **分类C**：缺项最多，需要最多的创造性修正，放在最后集中处理

### 4.2 分类A实现（140条）

#### 4.2.1 实现步骤

1. **公式解析器**：将研报伪代码转为AST（抽象语法树）
2. **依赖分析**：自动提取每个因子的依赖函数和窗口参数
3. **批量计算**：按依赖拓扑排序，批量计算
4. **输出验证**：检查形状、NA比例、数值范围

#### 4.2.2 验证步骤

```python
def validate_category_a(factor_values: dict):
    """分类A因子验证"""
    checks = {}
    for name, values in factor_values.items():
        checks[name] = {
            'shape': values.shape,                    # (T, N)
            'na_ratio': np.isnan(values).mean(),      # 应 < 0.3
            'inf_ratio': np.isinf(values).mean(),     # 应为 0
            'mean': np.nanmean(values),               # 应在合理范围
            'std': np.nanstd(values),                 # 应 > 0
            'min': np.nanmin(values),
            'max': np.nanmax(values),
        }
    return pd.DataFrame(checks).T
```

### 4.3 分类D实现（9条：Alpha26,35,49,50,51,87,158,173,191）

#### 4.3.1 处理原则

对每条因子进行代数简化后再实现，简化后的公式应与原公式数学等价。

| 因子 | 问题 | 简化方案 |
|------|------|---------|
| Alpha26 | `RANK` 嵌套 `TSRANK` 冗余 | 内层TSRANK已归一化，外层RANK等价于再次归一化，可简化为单层 |
| Alpha35 | `DECAYLINEAR` 嵌套 `TSRANK` | 保留，但标注冗余 |
| Alpha49 | `CORR` 中 `RANK` 与原始值混用 | 统一为RANK后计算CORR |
| Alpha50 | `SUM` 嵌套 `RANK` | RANK后SUM无意义，简化为直接SUM |
| Alpha51 | `RANK` 嵌套 `DECAYLINEAR` | 保留，标注 |
| Alpha87 | `MAX` 嵌套 `RANK` | RANK已归一化，MAX无意义，简化为直接取值 |
| Alpha158 | `STD` 嵌套 `RANK` | RANK后STD量纲固定，标注尺度敏感性 |
| Alpha173 | `RANK` 嵌套 `CORR` | CORR已无量纲，RANK仅做归一化 |
| Alpha191 | `CORR` 嵌套 `RANK` | 同Alpha49 |

### 4.4 分类B实现（20条：语义歧义修正版）

详见 [第7节](#7-分类b语义歧义推荐选择与理由)。

### 4.5 分类C实现（22条：缺项修正版）

详见 [第8节](#8-分类c22条因子专属缺项候选修正方案)。

---

## 5. 阶段3：因子检验

### 5.1 IC分析

```python
def compute_ic(factor_values, forward_returns, method='pearson'):
    """
    计算每个因子的IC序列
    
    Parameters:
    - factor_values: dict {factor_name: pd.DataFrame (T, N)}
    - forward_returns: pd.DataFrame (T, N), T+2日收益率
    - method: 'pearson' (IC) 或 'spearman' (Rank IC)
    
    Returns:
    - ic_summary: DataFrame with IC_mean, IC_std, IR, IC>0 ratio
    """
```

**输出指标**：
| 指标 | 定义 | 研报基准 |
|------|------|---------|
| IC均值 | mean(IC_t) | — |
| IC标准差 | std(IC_t) | — |
| IR (信息系数) | IC_mean / IC_std | — |
| IC>0比例 | P(IC_t > 0) | — |
| Rank IC均值 | mean(RankIC_t) | — |

### 5.2 分组收益

```python
def compute_group_returns(factor_values, forward_returns, n_groups=10):
    """
    按因子值分10组，计算每组等权收益
    
    Returns:
    - group_returns: (T, n_groups) 每组日收益
    - long_short: Top - Bottom 多空收益
    """
```

**输出**：
- 分组年化收益柱状图
- Top-Bottom多空累计收益曲线
- 多空最大回撤、夏普比率

### 5.3 显著性检验

```python
def ic_significance_test(ic_series):
    """
    Newey-West调整的t检验（滞后2期）
    H0: IC_mean = 0
    """
```

### 5.4 因子相关性分析

```python
def factor_correlation_matrix(factor_values):
    """
    计算因子截面平均相关性矩阵
    识别高相关因子簇（|ρ| > 0.7）
    """
```

**输出**：
- 191×191相关性矩阵热力图
- 高相关因子簇列表（用于后续因子合成时的去重）

---

## 6. 阶段4：策略复现

### 6.1 中性化处理

#### 6.1.1 行业中性化

```python
def industry_neutralize(factor_values, industry_dummy):
    """
    对每个截面做行业哑变量回归，取残差
    
    factor_residual = factor - industry_dummy @ beta_industry
    """
```

**行业分类**：申万一级行业（28个，2014版）

#### 6.1.2 风格中性化（BARRA CNE5）

```python
def style_neutralize(factor_values, style_factors):
    """
    对每个截面做风格因子回归，取残差
    
    BARRA CNE5风格因子：
    - SIZE: ln(总市值)
    - BETA: 历史Beta
    - MOMENTUM: 过去12个月动量
    - VOLATILITY: 历史波动率
    - GROWTH: 盈利增长率
    - VALUE: BP（市净率倒数）
    - LEVERAGE: 资产负债率
    - LIQUIDITY: 换手率
    - SIZENL: 非线性规模
    - EARNINGS_YIELD: 盈利率
    """
```

#### 6.1.3 正交标准化

```python
def orthogonalize_and_standardize(factor_values):
    """
    截面标准化：Z-score = (x - mean(x)) / std(x)
    使因子跨期可比
    """
```

### 6.2 因子合成

```python
def synthesize_factors(factor_values, method='equal_weight'):
    """
    合成方法：
    1. equal_weight: 所有因子等权
    2. ic_weight: 按过去60日IC均值加权
    3. ic_ir_weight: 按IC_IR加权
    4. max_ic_ir: 最优化权重（最大化合成因子IC_IR）
    """
```

**推荐**：先用等权验证框架正确性，再用IC_IR加权优化。

### 6.3 组合优化

```python
def optimize_portfolio(scores, constraints):
    """
    约束条件：
    - 个股权重上限：1%
    - 行业偏离：±3%
    - 风格偏离：±0.3σ
    - 换手率约束：单边20%
    - 做多约束：不允许做空
    
    目标函数：最大化组合得分（合成因子值）
    """
```

### 6.4 回测

```python
def run_backtest(weights, prices, tc=0.003):
    """
    回测参数：
    - 预测周期：2日
    - 成交价：次日开盘价
    - 交易成本：Tc=0.3%（单边）
    - 基准：中证500
    
    输出：
    - 年化超额收益
    - 最大回撤
    - 信息比率
    - 年化换手率
    - 月度胜率
    """
```

**研报基准对齐**：
| 指标 | 研报值 | 允许偏差 |
|------|--------|---------|
| 年化超额收益 | 50.2% | ±5% |
| 最大回撤 | 5.9% | ±2% |
| 信息比率 | 4.67 | ±0.5 |

---

## 7. 分类B：语义歧义——推荐选择与理由

### 7.1 MAX/MIN的逐元素vs滚动含义

**问题**：Alpha7,10,17,28,41,64,91,108,119,121,127,154,157,162,164中，`MAX(X, Y)` 是逐元素取大还是滚动窗口取大？

**推荐**：**逐元素取大**（`np.maximum(X, Y)`）。

**理由**：
1. 附录中 `MAX(S, n)` 已定义为滚动窗口最大值（单序列+窗口参数）
2. 当 `MAX` 接受两个序列参数时，语义自然变为逐元素比较
3. 若为滚动窗口，应写作 `MAX(X, n)` 而非 `MAX(X, Y)`
4. 逐元素取大在因子构建中更常见（如 `MAX(OPEN, CLOSE(-1))` 取开盘价与昨收的较大者）

**逐因子确认**：

| 因子 | MAX/MIN用法 | 解释 |
|------|------------|------|
| Alpha7 | `MAX(CLOSE-DELAY(CLOSE,1), 0)` | 逐元素：涨幅（负值截断为0） |
| Alpha10 | `MAX(MAX(HIGH-LOW, ABS(HIGH-DELAY(CLOSE,1))), ABS(LOW-DELAY(CLOSE,1)))` | 逐元素：真实波幅(TR) |
| Alpha17 | `MAX(OPEN, DELAY(CLOSE,1))` | 逐元素：开盘价与昨收较大者 |
| Alpha28 | `MAX(HIGH-DELAY(CLOSE,1), 0)` | 逐元素 |
| Alpha41 | `MAX(HIGH-LOW, ABS(CLOSE-OPEN))` | 逐元素 |
| Alpha64 | `MAX(MAX(HIGH-LOW, ABS(HIGH-DELAY(CLOSE,1))), ABS(LOW-DELAY(CLOSE,1)))` | 逐元素：同Alpha10 |
| Alpha91 | `MAX(CLOSE-DELAY(CLOSE,1), 0)` | 逐元素 |
| Alpha108 | `MAX(OPEN-DELAY(CLOSE,1), 0)` | 逐元素 |
| Alpha119 | `MAX(CLOSE-DELAY(CLOSE,5), 0)` | 逐元素 |
| Alpha121 | `MAX(CLOSE-DELAY(CLOSE,10), 0)` | 逐元素 |
| Alpha127 | `MAX(CLOSE-DELAY(CLOSE,20), 0)` | 逐元素 |
| Alpha154 | `MAX(CLOSE-DELAY(CLOSE,1), 0)` | 逐元素 |
| Alpha157 | `MAX(CLOSE-DELAY(CLOSE,5), 0)` | 逐元素 |
| Alpha162 | `MAX(CLOSE-DELAY(CLOSE,10), 0)` | 逐元素 |
| Alpha164 | `MAX(CLOSE-DELAY(CLOSE,20), 0)` | 逐元素 |

### 7.2 基准指数变量命名反了

**问题**：`BANCHMARKINDEXCLOSE` 实际定义为开盘价，`BANCHMARKINDEXOPEN` 实际定义为收盘价。

**推荐**：**按实际语义使用，不按变量名**。

即：
- 当因子需要基准指数开盘价时，使用 `BANCHMARKINDEXCLOSE`（研报中的变量名）
- 当因子需要基准指数收盘价时，使用 `BANCHMARKINDEXOPEN`（研报中的变量名）

**实现**：在代码中重命名为 `BENCHMARK_OPEN` 和 `BENCHMARK_CLOSE`，映射到正确的Wind字段。

### 7.3 LOWDAY定义与函数名冲突

**问题**：`LOWDAY(S, n)` 被定义为"最大值距今间隔"，与函数名（最低值日）矛盾。

**推荐**：**按函数名语义实现**（最低值距今间隔）。

**理由**：
1. 函数名 `LOWDAY` 明确暗示最低值
2. `HIGHDAY` 和 `LOWDAY` 应对称
3. 若按"最大值距今间隔"实现，则与 `HIGHDAY` 功能重复，`LOWDAY` 函数无存在意义
4. 研报中可能为笔误

### 7.4 DBM下降日开盘差方向退化

**问题**：`DBM` 定义为 `OPEN(t) - CLOSE(t-1)` 当 `OPEN(t) < CLOSE(t-1)`，结果始终为负。

**推荐**：**保持负值**。

**理由**：
1. 负值保留了方向信息（下跌缺口的大小和方向）
2. 后续因子中 `DBM` 通常与 `DTM` 配对使用（如 `DTM-DBM`），方向信息有意义
3. 若取绝对值会丢失方向

### 7.5 RANK/TSRANK数值范围

**问题**：RANK和TSRANK的数值范围影响后续幂运算和符号。

**推荐**：**归一化到 [0, 1]**。

**理由**：
1. 与 `TSRANK` 保持一致（TSRANK通常返回百分位）
2. 使不同因子的RANK结果可比
3. 幂运算（如 `RANK(...)^2`）在[0,1]范围内更稳定
4. 避免因股票数量变化导致的量纲漂移

---

## 8. 分类C：22条因子专属缺项——候选修正方案

### 8.1 Alpha10

**审计问题**：`MAX` 只有一个参数，而 `RANK` 多出参数5；括号归属不明确。

**修正**：按附录签名，`MAX` 需要两个参数。候选修正为对条件序列取5日极值后再 RANK：`RANK(MAX(MAX(HIGH-LOW, ABS(HIGH-DELAY(CLOSE,1))), ABS(LOW-DELAY(CLOSE,1))), 5))`。置信度：中。

### 8.2 Alpha21

**审计问题**：`REGBETA(A, SEQUENCE(n))` 缺第三参数（回归窗口）。

**修正**：`SEQUENCE(6)` 暗示 n=6，补为 `REGBETA(CLOSE, SEQUENCE(6), 6)`。置信度：中（属推断）。

### 8.3 Alpha22

**审计问题**：`SMEAN` 在附录中未定义。

**修正**：`SMEAN` → `SMA` 为候选别名修正。需同时补 SMA 的 n 和 m 参数，候选 n=10, m=2（与其他因子中 SMA 用法一致）。置信度：中。

### 8.4 Alpha23

**审计问题**：分子写成 `?STD(CLOSE:20),0`，三元条件的冒号与逗号位置不合法。

**修正**：与分母对应，候选为 `?STD(CLOSE,20):0`，即 STD(CLOSE,20) 存在时取其值，否则取0。置信度：中。

### 8.5 Alpha30

**审计问题**：`DELAY(CLOSE)` 缺滞后期；`REGRESI` 使用五参数多元形式，附录只定义三参数一元形式；`MKT` 与附录 `MKE` 不一致；三因子构造未给。

**修正**：
- `DELAY(CLOSE)` → `DELAY(CLOSE, 1)`
- REGRESI 需定义60日三因子多元回归（MKT/SMB/HML），取当期残差
- `MKE` → `MKT`（拼写修正）
- 三因子构建：全A股、市值中位数分组SMB、BM 30%/70%分位分组HML、等权、月度再平衡、无风险利率一年期定存日化

置信度：低（三因子构建细节均为假设）。

### 8.6 Alpha36

**审计问题**：`CORR` 缺第三参数（窗口），`RANK` 出现额外参数2；6、2的括号归属需核实。

**修正**：候选为 `RANK(SUM(CORR(RANK(VOLUME), RANK(VWAP), 6), 2))`，即6日CORR后取2日SUM再RANK。置信度：中（括号归属属推断）。

### 8.7 Alpha52

**审计问题**：分母使用未定义变量 `L`。

**修正**：`L` → `LOW`，较强别名推断。置信度：高。

### 8.8 Alpha54

**审计问题**：`STD(ABS(CLOSE-OPEN))` 缺窗口；同式 CORR 的10不能证明 STD 也取10。

**修正**：窗口无法从本研报唯一恢复。候选 n=20（该因子其他窗口的众数），但标注为不可验证假设。置信度：低。

### 8.9 Alpha78

**审计问题**：`MA` 在附录中未定义。

**修正**：`MA` → `MEAN`（候选别名），不改动分母的 CLOSE 变量。需同时确认 MA 的窗口参数。置信度：中。

### 8.10 Alpha111

**审计问题**：使用未定义变量 `VOL`。

**修正**：`VOL` → `VOLUME`，较强别名推断。置信度：高。

### 8.11 Alpha116

**审计问题**：`SEQUENCE` 没有调用参数，写成 `REGBETA(CLOSE, SEQUENCE, 20)`。

**修正**：`SEQUENCE` → `SEQUENCE(20)`，与回归窗口一致。置信度：中。

### 8.12 Alpha127

**审计问题**：外层 `MEAN` 缺窗口；两个 `MAX(CLOSE,12)` 的含义也有歧义。

**修正**：外层 MEAN 窗口无法从内层12唯一确认。候选 n=12（与内层一致），但标注为推断。MAX(CLOSE,12) 按分类B 7.1 逐元素处理。置信度：低。

### 8.13 Alpha131

**审计问题**：`DELAT` 未定义。

**修正**：`DELAT` → `DELTA`，明显拼写错误。置信度：高。

### 8.14 Alpha143

**审计问题**：`SELF` 只定义为前一日因子值，没有递推初值或重启规则；初值为0会导致全序列为0。

**修正**：定义 SELF 初值为首个有效因子值（非0），起始日为因子首次计算日。停牌后续接规则：前向填充最多5日，超过则重新初始化。不得擅自补收益率的+1。置信度：中。

### 8.15 Alpha146

**审计问题**：分母 `SMA(A,60)` 缺第三参数 m；内部 `r-(r-SMA(r,61,2))` 可抵消，是否为预期结构须核实。

**修正**：m 无法唯一确定。候选 m=2（与其他SMA一致），但内部相减可能抵消成另一个SMA，须核实预期结构。置信度：低。

### 8.16 Alpha147

**审计问题**：`REGBETA` 缺窗口 n；`SEQUENCE(12)` 暗示 n=12，但属推断。

**修正**：补为 `REGBETA(CLOSE, SEQUENCE(12), 12)`。置信度：中。

### 8.17 Alpha159

**审计问题**：`HGIH` 未定义；第三权重 6×24 与分母 6×12 不对应；分子 CLOSE-SUM(...) 的聚合位置可疑。

**修正**：`HGIH` → `HIGH`（拼写修正）。第三权重候选为 6×12（与分母对应），但不确定。CLOSE-SUM(...) 的聚合范围须核实。置信度：低。

### 8.18 Alpha165

**审计问题**：`SUMAC` 缺窗口；MAX/MIN 各只有一个参数；仅 MIN 项除以 STD，整体范围是否应除以 STD 未明确。

**修正**：须明确 SUMAC 窗口（候选48/24）、极值范围及整个极差的括号。48/24 只是候选窗口，不可当作已确认。置信度：低。

### 8.19 Alpha166

**审计问题**：分母有 `(20-2)(SUM(...))` 缺显式运算符；`(CLOSE/DELAY(CLOSE,1),20)^2` 不是合法的数值表达式；不能唯一恢复原意。

**修正**：不足以唯一恢复。若按偏度指标重构，必须标为新的候选版本，不能冒充原文。置信度：低。

### 8.20 Alpha181

**审计问题**：分母 `SUM` 缺窗口；分子将个股收益率偏离减去指数价格偏离的平方，指标结构需核实。

**修正**：需确认指数应为价格还是收益率、相减还是相乘、窗口与指数变量。使用中证500 CLOSE（见7.2）。暂缓严格实现，标为待核实版本。置信度：低。

### 8.21 Alpha183

**审计问题**：与 Alpha165 同类问题，均值及标准差窗口为24，但 SUMAC 与极值计算的窗口未给。

**修正**：候选窗口24，但 SUMAC 与极值计算的窗口不可确认。置信度：低。

### 8.22 Alpha190

**审计问题**：多处 `DELAY(CLOSE)` 缺滞后期；分子和分母的平方偏离中心相差2；LOG 定义域与零分母处理未给。

**修正**：`DELAY(CLOSE)` → `DELAY(CLOSE, 1)`。令 r=CLOSE/DELAY(CLOSE,1)-1，q=(CLOSE/DELAY(CLOSE,19))^(1/20)，g=q-1。第一个平方项=(r-g-2)^2，第二个平方项=(r-g)^2，并不对称。LOG 输入非正时返回 NaN。须逐项重写并核实。置信度：低。

### 8.23 缺项修正汇总

| 缺项类型 | 数量 | 修正策略 | 置信度 |
|---------|------|---------|--------|
| 拼写/别名修正 | 4条 | L→LOW, VOL→VOLUME, DELAT→DELTA, HGIH→HIGH, SMEAN→SMA, MA→MEAN | 高 |
| 缺窗口/参数补默认值 | 6条 | REGBETA补n, STD补n=20, MEAN补n, SEQUENCE补参数 | 中 |
| 表达式损坏修复 | 4条 | Alpha23三元条件, Alpha36括号, Alpha166运算符, Alpha190滞后 | 中~低 |
| 结构性缺项（需定义） | 4条 | Alpha30三因子回归, Alpha143 SELF初值, Alpha165/183 SUMAC窗口, Alpha181指数结构 | 低 |
| 不可唯一恢复 | 4条 | Alpha54/146/159/166 无法从原文确认，标为候选版本 | 低 |

---

## 9. 分类D：代数冗余/尺度敏感——处理方案

### 9.1 处理原则

1. **数学等价简化优先**：若能证明简化后与原公式数学等价，直接简化
2. **标注保留**：若简化可能改变数值结果（如浮点精度），保留原公式但标注
3. **尺度鲁棒化**：对尺度敏感的因子，在计算后做截面Z-score标准化

### 9.2 逐因子处理

| 因子 | 问题 | 处理方案 |
|------|------|---------|
| Alpha26 | RANK(RANK(...)) 冗余 | 简化为单层RANK，标注 |
| Alpha35 | DECAYLINEAR(TSRANK(...)) | 保留原公式，TSRANK已归一化故DECAYLINEAR稳定 |
| Alpha49 | CORR(RANK(X), RANK(Y)) | 保留，等价于Spearman相关 |
| Alpha50 | SUM(RANK(...)) | 简化为直接SUM，RANK单调变换不影响排序 |
| Alpha51 | RANK(DECAYLINEAR(...)) | 保留，DECAYLINEAR输出量纲依赖价格水平 |
| Alpha87 | MAX(RANK(...)) | 简化为直接取RANK值（RANK已归一化） |
| Alpha158 | STD(RANK(...)) | 保留但标注：RANK后STD量纲固定，截面可比性差 |
| Alpha173 | RANK(CORR(...)) | 保留，CORR已无量纲 |
| Alpha191 | CORR(RANK(X), RANK(Y)) | 同Alpha49，保留 |

---

## 10. 风险与假设清单

### 10.1 高置信度假设（几乎确定）

| # | 假设 | 置信度 | 影响范围 |
|---|------|--------|---------|
| H1 | 前复权 | 高 | 所有因子 |
| H2 | VWAP = AMOUNT/VOLUME | 高 | 使用VWAP的因子 |
| H3 | 窗口按交易日计数 | 高 | 所有滚动窗口 |
| H4 | STD ddof=1 | 高 | 所有使用STD的因子 |
| H5 | CORR为Pearson | 高 | 所有使用CORR的因子 |
| H6 | MAX(X,Y)为逐元素取大 | 高 | 15条因子 |
| H7 | RANK归一化到[0,1] | 高 | 所有使用RANK的因子 |
| H8 | 窗口包含当日 | 高 | 所有滚动窗口 |

### 10.2 中置信度假设（合理推断）

| # | 假设 | 置信度 | 影响范围 |
|---|------|--------|---------|
| M1 | SMA初值为S_raw(0) | 中 | 使用SMA的因子 |
| M2 | SMA同日输出（非滞后） | 中 | 使用SMA的因子 |
| M3 | WMA有限窗口归一化 | 中 | 使用WMA的因子 |
| M4 | DECAYLINEAR当日权重=n | 中 | 使用DECAYLINEAR的因子 |
| M5 | REGBETA/REGRESI含截距 | 中 | 使用回归的因子 |
| M6 | 零方差CORR返回0 | 中 | 所有CORR因子 |
| M7 | 停牌前向填充5日 | 中 | 所有因子 |
| M8 | 分类C缺项默认n=20 | 中 | 16条因子 |
| M9 | LOWDAY按函数名语义 | 中 | 使用LOWDAY的因子 |
| M10 | DBM保持负值 | 中 | 使用DBM的因子 |

### 10.3 低置信度假设（需验证）

| # | 假设 | 置信度 | 影响范围 |
|---|------|--------|---------|
| L1 | 三因子构建方式（等权/市值加权） | 低 | Alpha30 |
| L2 | 行业分类版本（申万2014 vs 2021） | 低 | 中性化 |
| L3 | BARRA因子具体构建方式 | 低 | 风格中性化 |
| L4 | 组合优化具体约束参数 | 低 | 回测结果 |
| L5 | 研报回测的成交价假设（开盘价vs VWAP） | 低 | 回测结果 |

### 10.4 关键风险

| 风险 | 严重程度 | 缓解措施 |
|------|---------|---------|
| 研报结果不可复现（选择性报告/过拟合） | 高 | 分因子检验IC，识别真正有效的因子子集 |
| 191因子高度相关，合成后信息增量有限 | 高 | 相关性分析后筛选独立因子子集 |
| 2017年后市场结构变化导致因子衰减 | 中 | 扩展回测期至2024年，评估衰减程度 |
| 交易成本假设过于乐观（0.3%单边） | 中 | 敏感性分析：Tc=0.1%/0.3%/0.5% |
| 停牌/涨跌停处理影响小盘股因子值 | 中 | 对比含/不含涨跌停过滤的IC差异 |
| 前复权vs后复权影响价格水平因子 | 低 | 敏感性分析 |

---

## 11. 验证与对齐检查点

### 11.1 阶段0检查点

- [ ] 数据完整性：全A股OHLCV覆盖2010-01至2017-04，缺失率<5%
- [ ] 复权正确性：前复权后CLOSE序列无除权跳空
- [ ] 股票池正确性：ST/停牌/涨跌停过滤后，日均股票数在2000-3500之间
- [ ] 三因子合理性：MKT/SMB/HML均值与A股历史特征一致
- [ ] 预热长度：500个交易日覆盖所有因子最大窗口

### 11.2 阶段1检查点

- [ ] 每个基础函数通过单元测试（覆盖率>90%）
- [ ] 面板向量化实现与逐股循环实现结果一致（抽样验证）
- [ ] 边界条件测试通过（NaN输入、零方差、单期、空股票池）
- [ ] 性能基准：全A股500只股票×1500日，单函数计算<1秒

### 11.3 阶段2检查点

- [ ] 分类A 140条因子全部成功计算，无运行时错误
- [ ] 因子NA比例<30%（预热期除外）
- [ ] 因子截面均值时间序列无明显趋势（排除数据泄露）
- [ ] 分类D 9条因子简化后与原公式数值差异<1e-10
- [ ] 分类B 20条因子语义修正后，因子值在合理范围
- [ ] 分类C 22条因子修正后，因子值分布与其他因子可比

### 11.4 阶段3检查点

- [ ] 至少30%的因子IC均值绝对值>0.02（显著）
- [ ] Rank IC均值分布近似正态，中心在0附近
- [ ] Top-Bottom分组收益单调递增（或递减）
- [ ] 因子相关性矩阵无不合理的高相关簇（|ρ|>0.95的因子对<10%）
- [ ] Newey-West t检验：至少20%的因子在5%水平显著

### 11.5 阶段4检查点

- [ ] 中性化后因子与行业/风格的相关性<0.05
- [ ] 合成因子IC_IR > 单因子IC_IR中位数
- [ ] 回测年化超额收益在研报值±5%范围内（45.2%-55.2%）
- [ ] 最大回撤在研报值±2%范围内（3.9%-7.9%）
- [ ] 信息比率在研报值±0.5范围内（4.17-5.17）
- [ ] 年化换手率合理（<50倍）

### 11.6 最终对齐检查

- [ ] 所有191个因子均可追溯到研报附录公式
- [ ] 所有假设和修正均有文档记录
- [ ] 回测结果与研报的偏差有合理解释
- [ ] 代码可复现（一键运行脚本）

---

## 12. 附录

### 12.1 因子分类完整清单

| 分类 | 数量 | Alpha编号 |
|------|------|-----------|
| A | 140 | 1-9, 11-16, 18-20, 24-25, 27, 29, 31-34, 37-40, 42-48, 53, 55-63, 65-77, 79-86, 88-90, 92-107, 109-110, 112-115, 117-118, 120, 122-126, 128-130, 132-142, 144-145, 148-153, 155-156, 160-161, 163, 167-172, 174-180, 182, 184-189 |
| B | 20 | 7, 10, 17, 28, 41, 64, 91, 108, 119, 121, 127, 154, 157, 162, 164, +5条DBM/LOWDAY相关 |
| C | 22 | 10, 21, 22, 23, 30, 36, 52, 54, 78, 111, 116, 127, 131, 143, 146, 147, 159, 165, 166, 181, 183, 190 |
| D | 9 | 26, 35, 49, 50, 51, 87, 158, 173, 191 |

### 12.2 关键时间节点

| 里程碑 | 预计耗时 | 累计 |
|--------|---------|------|
| 阶段0：数据准备 | 1周 | 1周 |
| 阶段1：函数库 | 2周 | 3周 |
| 阶段2A：分类A因子 | 1周 | 4周 |
| 阶段2B-D：分类B/C/D因子 | 2周 | 6周 |
| 阶段3：因子检验 | 1周 | 7周 |
| 阶段4：策略复现 | 2周 | 9周 |
| 文档与对齐 | 1周 | 10周 |

### 12.3 参考文献

1. 国泰君安证券. 基于短周期价量特征的多因子选股体系. 2017年6月.
2. Fama, E.F. & French, K.R. (1993). Common risk factors in the returns on stocks and bonds. *Journal of Financial Economics*, 33(1), 3-56.
3. Menchero, J. et al. (2011). The Barra China Equity Model (CNE5). MSCI Research.
4. Newey, W.K. & West, K.D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703-708.

---

> **文档状态**: 待执行  
> **下一步**: 阶段0——数据获取与预处理  
> **维护者**: 量化因子工程团队  
> **最后更新**: 2026-10-12