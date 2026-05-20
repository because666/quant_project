# 回测框架完善（第二轮）Spec

## Why
上一轮修复了停牌/涨跌停/基准净值/统计检验函数等基础问题，但回测框架仍存在以下影响论文实验的缺口：回测结果缺少收益率与超额收益率列、缺少Calmar/Sortino/IC/ICIR/MAP等论文常用指标、统计检验与分年度分析未集成到回测导出流程、缺少多Top N参数敏感性对比、缺少在线触发回测的API端点、高级指标函数无测试覆盖。

## What Changes
- 回测结果DataFrame新增 `weekly_return`（周度收益率）和 `excess_return_hs300`/`excess_return_zz500`（超额收益率）列
- metrics.py 新增 Calmar比率、Sortino比率、IC/ICIR、MAP 指标函数
- 回测导出流程集成统计检验、分年度分析、多空分析
- 新增多Top N对比回测函数 `run_multi_topn_backtest`
- 新增 `POST /api/v1/backtest/run` 在线触发回测API端点
- 补全 bootstrap_metric / paired_significance_test / metrics_by_year / quantile_portfolio_returns 的单元测试
- **BREAKING**: `run_backtest` 返回的DataFrame新增3列；`compute_backtest_metrics` 返回字典新增多个指标键

## Impact
- Affected specs: 回测引擎核心逻辑、指标计算模块、API端点
- Affected code: `src/backtest.py`（收益率列、多Top N对比、导出集成）、`src/metrics.py`（新增指标函数）、`src/api/v1/backtest.py`（新增POST端点）、`tests/test_metrics.py`（测试补全）、`tests/test_backtest.py`（测试补全）

## ADDED Requirements

### Requirement: 周度收益率与超额收益率
系统 SHALL 在回测结果DataFrame中自动计算并添加周度收益率列和相对基准的超额收益率列。

#### Scenario: 回测结果包含收益率列
- **WHEN** 回测完成并返回结果DataFrame
- **THEN** 结果中包含 `weekly_return` 列（周度收益率，首行设为0）、`excess_return_hs300` 列（策略周度收益减去沪深300同期收益）、`excess_return_zz500` 列（策略周度收益减去中证500同期收益）

#### Scenario: 基准数据缺失时超额收益为NaN
- **WHEN** 基准净值获取失败（benchmark列为NaN）
- **THEN** 超额收益率列填充为NaN，不影响其他列计算

### Requirement: Calmar比率
系统 SHALL 提供Calmar比率计算函数，即年化收益率除以最大回撤。

#### Scenario: 计算Calmar比率
- **WHEN** 用户调用 `calmar_ratio(nav_series)`
- **THEN** 返回年化收益率与最大回撤的比值；最大回撤为0时返回NaN

### Requirement: Sortino比率
系统 SHALL 提供Sortino比率计算函数，即年化超额收益除以下行标准差。

#### Scenario: 计算Sortino比率
- **WHEN** 用户调用 `sortino_ratio(nav_series, risk_free_rate=0.03)`
- **THEN** 返回基于下行标准差的年化风险调整收益；无下行波动时返回NaN

### Requirement: IC与ICIR计算
系统 SHALL 提供因子IC（信息系数）和ICIR（信息比率）计算函数，用于评估因子预测能力。

#### Scenario: 计算单因子IC序列
- **WHEN** 用户调用 `factor_ic(factor_values, return_values, groupby=None)`
- **THEN** 返回每期（或全局）的Spearman秩相关系数序列；若提供groupby列则按期分组计算

#### Scenario: 计算ICIR
- **WHEN** 用户调用 `factor_icir(ic_series)`
- **THEN** 返回IC均值除以IC标准差

### Requirement: MAP指标
系统 SHALL 提供MAP（Mean Average Precision）排序质量指标计算函数。

#### Scenario: 计算MAP
- **WHEN** 用户调用 `mean_average_precision(y_true_groups, y_score_groups)`
- **THEN** 返回所有query的AP均值，y_true和y_score按query分组

### Requirement: 统计检验集成到回测导出
系统 SHALL 在 `compute_backtest_metrics` 中集成Bootstrap置信区间和配对显著性检验结果。

#### Scenario: 扩展指标包含统计检验
- **WHEN** 用户调用 `compute_backtest_metrics(result_df, weekly_df, extended=True)`
- **THEN** 返回字典中包含 `sharpe_ci`（夏普比率Bootstrap置信区间）和 `annualized_return_ci`（年化收益Bootstrap置信区间）

### Requirement: 多Top N对比回测
系统 SHALL 提供多Top N参数对比回测函数，一次性跑多个选股数量并汇总对比结果。

#### Scenario: 多Top N对比
- **WHEN** 用户调用 `run_multi_topn_backtest(top_n_list=[5, 10, 20, 30])`
- **THEN** 返回每个Top N的回测结果和指标汇总表，便于参数敏感性分析

### Requirement: 在线触发回测API
系统 SHALL 提供POST端点用于在线触发回测，支持参数配置。

#### Scenario: 触发回测
- **WHEN** 前端发送 `POST /api/v1/backtest/run` 请求，body包含 `model_type`, `top_n`, `initial_capital` 等参数
- **THEN** 后端执行回测、落库、导出JSON，返回回测结果ID和核心指标

#### Scenario: 回测执行超时或失败
- **WHEN** 回测执行过程中发生异常
- **THEN** 返回HTTP 500，body包含错误信息，不中断服务

### Requirement: 高级指标函数测试覆盖
系统 SHALL 为 `bootstrap_metric`、`paired_significance_test`、`metrics_by_year`、`quantile_portfolio_returns`、`calmar_ratio`、`sortino_ratio`、`factor_ic`、`factor_icir`、`mean_average_precision` 编写单元测试。

#### Scenario: 测试覆盖
- **WHEN** 运行 `pytest tests/test_metrics.py`
- **THEN** 所有新增函数的测试用例通过，覆盖正常流程、边界情况

## MODIFIED Requirements

### Requirement: compute_backtest_metrics返回值
返回字典新增以下键：
- `calmar_ratio`: Calmar比率
- `sortino_ratio`: Sortino比率
- `sharpe_ci`: 夏普比率Bootstrap置信区间（extended=True时）
- `annualized_return_ci`: 年化收益Bootstrap置信区间（extended=True时）

### Requirement: run_backtest返回值
返回的DataFrame新增以下列：
- `weekly_return`: 周度收益率（首行=0）
- `excess_return_hs300`: 相对沪深300超额收益率
- `excess_return_zz500`: 相对中证500超额收益率

## REMOVED Requirements
无移除项。
