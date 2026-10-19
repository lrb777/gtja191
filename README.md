# GTJA Alpha191 因子复现

复现国泰君安《基于短周期价量特征的多因子选股体系》（2017年6月）中的191个Alpha因子。

## 目录结构

```
gtja191/
├── config/settings.py          # 全局配置
├── data/
│   ├── raw/                    # tushare原始数据
│   └── processed/              # 复权对齐后数据
├── src/
│   ├── data/                   # 数据获取与预处理
│   ├── lib/                    # 基础函数库（30个附录函数）
│   ├── factors/                # 191因子实现
│   ├── analysis/               # IC/分组/相关性分析
│   └── strategy/               # 中性化/合成/回测
├── tests/                      # 单元测试
├── scripts/                    # 流程脚本
└── output/                     # 图表与报告
```

## 执行顺序

1. `scripts/step0_fetch_data.py` — tushare数据获取
2. `scripts/step1_preprocess.py` — 复权、清洗、对齐
3. `src/lib/base_functions.py` — 函数库实现
4. `src/factors/` — 191因子分层实现（A→D→B→C）
5. `scripts/step2_compute_factors.py` — 批量计算
6. `scripts/step3_evaluate.py` — IC/分组/相关性检验

## 数据源

- tushare pro API
- 全A股，前复权
- 基准：中证500 (000905.SH)
- 时间：2010-01-01 ~ 2017-04-30

## 文档

- `gtja_alpha191_formula_audit.md` — 公式完备性检查
- `gtja_alpha191_implementation_plan.md` — 实现计划
