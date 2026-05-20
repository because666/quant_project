# Tasks

- [x] Task 1: 修复停牌股票处理逻辑
  - [x] SubTask 1.1: 在 `BacktestEngine` 中新增 `last_known_price: dict[str, float]` 属性，用于缓存每只股票的最后已知收盘价
  - [x] SubTask 1.2: 修改 `run_backtest` 中第328-330行的持仓清理逻辑：不在截面中的股票不再删除，而是使用 `last_known_price` 估值
  - [x] SubTask 1.3: 修改 `_portfolio_market_value` 函数，支持对不在截面中的持仓使用 `last_known_price` 估值
  - [x] SubTask 1.4: 编写单元测试：验证停牌股票持仓保留、市值不为零、复牌后恢复正常估值

- [x] Task 2: 简化涨跌停处理
  - [x] SubTask 2.1: 在 `BacktestEngine.__init__` 中新增 `enable_limit_price: bool = False` 参数
  - [x] SubTask 2.2: 修改 `load_weekly_data` 中的涨跌停标记逻辑：当 `enable_limit_price=False` 时，`buy_blocked_limit_up` 和 `sell_blocked_limit_down` 全部设为 `False`
  - [x] SubTask 2.3: 修改 `_select_target_codes` 函数：当涨跌停禁用时跳过涨停过滤
  - [x] SubTask 2.4: 编写单元测试：验证默认模式下无涨跌停约束、启用模式下保留原有逻辑

- [x] Task 3: 添加基准净值曲线
  - [x] SubTask 3.1: 新增 `_load_benchmark_nav` 函数，通过akshare获取沪深300和中证500周频数据，计算累计净值
  - [x] SubTask 3.2: 在 `run_backtest` 返回结果中合并基准净值列 `benchmark_hs300` 和 `benchmark_zz500`
  - [x] SubTask 3.3: 处理基准数据获取失败的降级逻辑（填充NaN、记录警告日志）
  - [x] SubTask 3.4: 编写单元测试：验证基准净值列存在、起始值为1.0、长度与回测结果一致

- [x] Task 4: 添加统计检验函数到 metrics.py
  - [x] SubTask 4.1: 实现 `bootstrap_metric` 函数：对任意策略指标构建Bootstrap置信区间
  - [x] SubTask 4.2: 实现 `paired_significance_test` 函数：配对t检验/Wilcoxon检验
  - [x] SubTask 4.3: 编写单元测试：验证Bootstrap置信区间包含点估计、配对检验对相同分布返回不显著

- [x] Task 5: 添加分年度和多空分析函数到 metrics.py
  - [x] SubTask 5.1: 实现 `metrics_by_year` 函数：按自然年拆分净值序列，计算每年的核心指标
  - [x] SubTask 5.2: 实现 `quantile_portfolio_returns` 函数：按得分分位数分组计算组合收益和多空收益
  - [x] SubTask 5.3: 编写单元测试：验证分年度指标计算正确、多空组合收益=Q1-Q5

- [x] Task 6: 集成验证
  - [x] SubTask 6.1: 运行完整回测（LightGBM + XGBoost），确认停牌修复后净值曲线无断崖
  - [x] SubTask 6.2: 确认基准净值曲线正常显示
  - [x] SubTask 6.3: 确认统计检验函数可正常调用
  - [x] SubTask 6.4: 运行全部单元测试，确认通过

# Task Dependencies
- [Task 2] depends on [Task 1] (停牌修复后再改涨跌停，避免同时修改核心逻辑)
- [Task 3] depends on [Task 1] (基准对比需要停牌修复后的正确净值)
- [Task 4] and [Task 5] are independent of each other and of Tasks 1-3
- [Task 6] depends on [Task 1, Task 2, Task 3, Task 4, Task 5]
