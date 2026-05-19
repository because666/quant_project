# 回测框架完善 Spec

## Why
当前回测引擎存在3个影响结果正确性的缺陷（停牌市值归零、涨跌停周频误判、缺少基准对比），以及2个影响学术可信度的缺失（缺少统计检验、缺少分年度/多空分析）。修复后回测结果才能用于论文实验。

## What Changes
- 修复停牌股票处理：不在截面中的持仓股票保留持仓，使用最后已知收盘价估值，而非删除（市值归零）
- 简化涨跌停处理：移除基于周频数据的涨跌停判断（逻辑不正确），改为可选的日频涨跌停检查；默认模式下不施加涨跌停约束，在论文中声明简化假设
- 添加基准净值曲线：在回测结果中增加沪深300和中证500的同期净值
- 添加统计检验模块：Bootstrap置信区间、配对显著性检验
- 添加分年度指标计算和多空组合收益计算
- **BREAKING**: `run_backtest` 返回的DataFrame新增 `benchmark_hs300`、`benchmark_zz500` 列；涨跌停默认关闭

## Impact
- Affected specs: 回测引擎核心逻辑、指标计算模块
- Affected code: `src/backtest.py`（停牌处理、涨跌停逻辑、基准净值）、`src/metrics.py`（新增统计检验函数、分年度计算、多空收益）

## ADDED Requirements

### Requirement: 停牌股票保留持仓估值
系统 SHALL 在回测过程中，当持仓股票不在当前截面数据中时，保留该持仓并使用最后已知收盘价进行估值，而非删除持仓（市值归零）。

#### Scenario: 持仓股票停牌
- **WHEN** 某只持仓股票在调仓日截面数据中不存在（可能停牌或退市）
- **THEN** 系统保留该股票持仓，使用其最后一次出现的收盘价进行估值，持仓市值不归零

#### Scenario: 停牌股票复牌
- **WHEN** 之前不在截面中的股票重新出现在截面中
- **THEN** 系统恢复使用当前截面收盘价估值，该股票可正常参与后续调仓逻辑

### Requirement: 涨跌停处理简化
系统 SHALL 默认不施加涨跌停约束（因为周频数据无法正确判断日频涨跌停），通过参数 `enable_limit_price` 控制是否启用涨跌停检查，默认为 `False`。

#### Scenario: 默认模式（不施加涨跌停约束）
- **WHEN** 用户运行回测且未显式启用涨跌停
- **THEN** 所有股票均可正常买卖，不受涨跌停限制

#### Scenario: 显式启用涨跌停
- **WHEN** 用户设置 `enable_limit_price=True`
- **THEN** 系统使用周频收盘价与前周收盘价比较进行涨跌停判断（保留现有逻辑，但标记为简化实现）

### Requirement: 基准净值曲线
系统 SHALL 在回测结果中自动添加沪深300和中证500指数的同期净值曲线，用于与策略净值对比。

#### Scenario: 回测结果包含基准
- **WHEN** 回测完成并返回结果DataFrame
- **THEN** 结果中包含 `benchmark_hs300` 和 `benchmark_zz500` 两列，分别为沪深300和中证500的累计净值（起始净值=1.0）

#### Scenario: 基准数据获取失败
- **WHEN** 通过akshare获取基准指数数据失败（网络问题等）
- **THEN** 基准列填充为NaN，回测不中断，记录警告日志

### Requirement: Bootstrap置信区间
系统 SHALL 提供对策略核心指标（年化收益、夏普比率）的Bootstrap置信区间计算功能。

#### Scenario: 计算夏普比率置信区间
- **WHEN** 用户调用 `bootstrap_metric(returns, sharpe_ratio_fn, n_bootstrap=5000)`
- **THEN** 返回包含 `point`（点估计）、`lower`（下界）、`upper`（上界）的字典，置信水平默认95%

### Requirement: 配对显著性检验
系统 SHALL 提供两种方法的周度收益差异显著性检验（配对t检验和Wilcoxon符号秩检验），自动选择正态性假设下的检验方法。

#### Scenario: 检验两种方法收益差异
- **WHEN** 用户调用 `paired_significance_test(returns_a, returns_b)`
- **THEN** 返回包含 `t_stat`、`p_value`、`significant`、`method` 的字典

### Requirement: 分年度指标计算
系统 SHALL 支持按自然年拆分回测结果，计算每年的年化收益、夏普比率、最大回撤。

#### Scenario: 分年度分析
- **WHEN** 用户调用 `metrics_by_year(nav_series)`
- **THEN** 返回按年份索引的DataFrame，包含每年的年化收益、夏普比率、最大回撤

### Requirement: 多空组合收益计算
系统 SHALL 支持按模型得分将股票分为N个分位数组，计算每组的等权组合收益和多空组合收益。

#### Scenario: 五分位多空组合
- **WHEN** 用户调用 `quantile_portfolio_returns(scores_df, n_quantiles=5)`
- **THEN** 返回包含Q1-Q5各组收益率序列和LS（Q1-Q5）多空收益率序列的字典

## MODIFIED Requirements

### Requirement: BacktestEngine构造函数
新增 `enable_limit_price: bool = False` 参数，控制是否启用涨跌停约束。默认关闭。

### Requirement: run_backtest返回值
返回的DataFrame新增以下列：
- `benchmark_hs300`: 沪深300同期累计净值
- `benchmark_zz500`: 中证500同期累计净值

## REMOVED Requirements

### Requirement: 默认启用涨跌停约束
**Reason**: 周频数据无法正确判断日频涨跌停现象，当前实现逻辑有根本性缺陷
**Migration**: 通过 `enable_limit_price=True` 参数可恢复原有涨跌停逻辑
