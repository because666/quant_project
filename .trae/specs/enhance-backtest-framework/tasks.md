# Tasks

- [x] Task 1: 回测结果新增收益率与超额收益率列
  - [x] SubTask 1.1: 在 `run_backtest` 返回结果前，计算 `weekly_return` 列（total_value逐行pct_change，首行设为0）
  - [x] SubTask 1.2: 计算 `excess_return_hs300` 列（weekly_return - benchmark_hs300的pct_change）
  - [x] SubTask 1.3: 计算 `excess_return_zz500` 列（weekly_return - benchmark_zz500的pct_change）
  - [x] SubTask 1.4: 处理基准缺失时超额收益填充NaN的逻辑
  - [x] SubTask 1.5: 编写单元测试验证收益率列存在、首行为0、超额收益计算正确

- [x] Task 2: 新增Calmar比率和Sortino比率函数
  - [x] SubTask 2.1: 实现 `calmar_ratio(nav_series)` 函数：年化收益 / 最大回撤
  - [x] SubTask 2.2: 实现 `sortino_ratio(nav_series, risk_free_rate)` 函数：年化超额收益 / 下行标准差 * sqrt(252)
  - [x] SubTask 2.3: 在 `aggregate_metrics` 中集成 calmar_ratio 和 sortino_ratio
  - [x] SubTask 2.4: 编写单元测试：Calmar比率正常计算、最大回撤为0时返回NaN；Sortino比率正常计算、无下行波动时返回NaN

- [x] Task 3: 新增IC/ICIR和MAP指标函数
  - [x] SubTask 3.1: 实现 `factor_ic(factor_values, return_values, groupby_col=None)` 函数：按期分组计算Spearman秩相关系数
  - [x] SubTask 3.2: 实现 `factor_icir(ic_series)` 函数：IC均值 / IC标准差
  - [x] SubTask 3.3: 实现 `mean_average_precision(y_true_groups, y_score_groups)` 函数：多query AP均值
  - [x] SubTask 3.4: 编写单元测试：IC计算正确、ICIR正常、MAP与手工计算一致

- [x] Task 4: 统计检验集成到回测导出流程
  - [x] SubTask 4.1: 修改 `compute_backtest_metrics` 函数，当 `extended=True` 时调用 `bootstrap_metric` 计算夏普比率和年化收益的置信区间
  - [x] SubTask 4.2: 在返回字典中添加 `sharpe_ci` 和 `annualized_return_ci` 键
  - [x] SubTask 4.3: 编写单元测试验证扩展指标包含置信区间

- [x] Task 5: 多Top N对比回测函数
  - [x] SubTask 5.1: 实现 `run_multi_topn_backtest(top_n_list, **kwargs)` 函数：遍历多个Top N值，调用BacktestEngine执行回测
  - [x] SubTask 5.2: 汇总各Top N的指标为对比表（DataFrame格式）
  - [x] SubTask 5.3: 编写单元测试验证多Top N对比结果结构正确

- [x] Task 6: 在线触发回测API端点
  - [x] SubTask 6.1: 在 `src/api/v1/backtest.py` 中新增 `POST /backtest/run` 端点，接收 model_type/top_n/initial_capital 等参数
  - [x] SubTask 6.2: 端点内部调用 `run_backtest_and_export`，返回结果ID和核心指标
  - [x] SubTask 6.3: 添加异常处理：回测失败返回HTTP 500和错误信息
  - [x] SubTask 6.4: 编写API测试验证端点响应格式和错误处理

- [x] Task 7: 补全高级指标函数的单元测试
  - [x] SubTask 7.1: 为 `bootstrap_metric` 编写测试：置信区间包含点估计、不同置信水平
  - [x] SubTask 7.2: 为 `paired_significance_test` 编写测试：相同分布不显著、不同分布显著
  - [x] SubTask 7.3: 为 `metrics_by_year` 编写测试：分年度指标正确、空输入返回空DataFrame
  - [x] SubTask 7.4: 为 `quantile_portfolio_returns` 编写测试：分组收益正确、多空收益=Q1-Q5

- [x] Task 8: 集成验证
  - [x] SubTask 8.1: 运行完整回测，确认新增收益率列和超额收益率列正常
  - [x] SubTask 8.2: 确认Calmar/Sortino/IC/ICIR/MAP函数可正常调用
  - [x] SubTask 8.3: 确认多Top N对比函数正常执行
  - [x] SubTask 8.4: 确认POST /backtest/run端点可正常触发回测
  - [x] SubTask 8.5: 运行全部单元测试，确认通过

# Task Dependencies
- [Task 2] depends on [Task 1]（Calmar需要年化收益和最大回撤，Sortino需要收益率序列）
- [Task 4] depends on [Task 2]（集成Bootstrap需要指标函数就绪）
- [Task 5] depends on [Task 1]（多Top N对比需要收益率列）
- [Task 6] depends on [Task 4]（API端点需要完整的指标计算）
- [Task 7] depends on [Task 2, Task 3]（测试需要新函数已实现）
- [Task 8] depends on [Task 1, Task 2, Task 3, Task 4, Task 5, Task 6, Task 7]
