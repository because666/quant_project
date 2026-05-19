# 修复项目核心问题 Spec

## Why
实验4和统计检验揭示了多个严重问题：模型排序方向反转导致多空组合收益为负、实验间结果不一致（B-LGBM与E1a-LGBM结果完全相同）、夏普比率计算口径不统一、DSR检验全部不通过、代码质量缺陷（换手率全0、持有期分析有误、global变量等）。这些问题直接影响论文结论的可信度，必须逐一修复。

## What Changes
- 修复模型排序方向问题：在分位数分析中支持vol_penalty调整分数，验证模型原始排序能力
- 修复统计检验中E1a-LGBM与B-LGBM结果完全相同的问题
- 统一夏普比率计算口径：所有实验和统计检验使用同一计算函数
- 修复DSR计算中的异常值（DSR接近0而非预期的0.4~0.9区间）
- 修复换手率全0问题：敏感性分析中添加extended=True
- 修复持有期敏感性分析实现有误的问题
- 移除global变量，改为函数参数传递
- 重新运行所有受影响的实验和统计检验

## Impact
- Affected specs: experiment1-loss-function, experiment2-fusion, experiment4, statistical_tests
- Affected code: scripts/run_statistical_tests.py, scripts/run_experiment4.py, src/metrics.py, src/backtest.py

## ADDED Requirements

### Requirement: 分位数分析支持vol_penalty调整
系统 SHALL 在 `quantile_portfolio_returns` 函数中支持 `vol_penalty` 参数，使分位数分组基于调整后分数（score - vol_penalty * volatility）而非原始分数，与回测引擎选股逻辑一致。

#### Scenario: vol_penalty分位数分析
- **WHEN** 调用 `quantile_portfolio_returns(df, n_quantiles=5, vol_penalty=1.0)`
- **THEN** Q1对应调整后分数最高的股票，多空组合收益应与回测策略收益方向一致

### Requirement: 统计检验中模型加载校验
系统 SHALL 在统计检验脚本中为每个方法输出模型文件的SHA256哈希前8位，确保加载了正确的模型文件。

#### Scenario: 模型加载校验
- **WHEN** 统计检验脚本加载模型文件
- **THEN** 输出日志中包含模型文件路径和SHA256哈希前8位，B-LGBM和E1a-LGBM的哈希必须不同

### Requirement: 统一夏普比率计算口径
系统 SHALL 在所有实验脚本和统计检验中使用 `compute_backtest_metrics` 函数计算夏普比率，禁止直接用周频收益率手动计算。

#### Scenario: 夏普比率口径一致
- **WHEN** 同一方法在不同实验/统计检验中计算夏普比率
- **THEN** 结果差异不超过0.01

### Requirement: 修复DSR计算
系统 SHALL 修复DSR计算中导致结果接近0的问题，确保使用正确的年化夏普比率和观测数。

#### Scenario: DSR计算正确
- **WHEN** 夏普比率为0.71、n_obs=126、n_trials=20
- **THEN** DSR应在0.3~0.9区间内，而非接近0

### Requirement: 修复换手率全0问题
系统 SHALL 在敏感性分析中调用 `compute_backtest_metrics` 时传入 `extended=True`，确保换手率被正确计算。

#### Scenario: 换手率非零
- **WHEN** 运行Top N敏感性分析
- **THEN** 所有方法的turnover_rate > 0

### Requirement: 修复持有期敏感性分析
系统 SHALL 在回测引擎层面支持 `rebalance_freq` 参数（调仓频率，单位周），实现真正的多周持有逻辑，而非简单降采样。

#### Scenario: 多周持有
- **WHEN** 设置 rebalance_freq=2（2周调仓一次）
- **THEN** 回测引擎每2周执行一次选股和调仓，中间周保持持仓不变

### Requirement: 移除global变量
系统 SHALL 将 `run_statistical_tests.py` 中的 `global n_trials` 改为函数参数传递。

#### Scenario: 无global变量
- **WHEN** 运行统计检验脚本
- **THEN** 不使用任何global语句

## MODIFIED Requirements

### Requirement: 重新运行受影响实验
统计检验和实验4 SHALL 在修复上述问题后重新运行，确保结果一致且正确。
