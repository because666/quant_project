# Tasks

- [x] Task 1: 修复分位数分析支持vol_penalty调整
  - [x] 1.1: 在 `src/metrics.py` 的 `quantile_portfolio_returns` 函数中增加 `vol_penalty` 参数和 `volatility_col` 参数
  - [x] 1.2: 当 `vol_penalty > 0` 时，使用 `score - vol_penalty * volatility` 作为分组依据
  - [x] 1.3: 在 `scripts/run_experiment4.py` 和 `scripts/run_statistical_tests.py` 中传入正确的 `vol_penalty` 参数
  - [x] 1.4: 验证修复后多空组合收益方向与回测策略一致 ✅ LS=8.90%~13.00%均为正

- [x] Task 2: 修复统计检验中E1a-LGBM与B-LGBM结果完全相同的问题
  - [x] 2.1: 在 `run_statistical_tests.py` 的 `run_all_backtests` 函数中添加模型文件SHA256校验日志
  - [x] 2.2: 根因：单模型方法未传入custom_predictor，BacktestEngine总是加载默认模型。已修复所有脚本传入custom_predictor
  - [x] 2.3: 验证修复后B-LGBM和E1a-LGBM的回测结果不同 ✅ 16.55% vs 12.60%

- [x] Task 3: 统一夏普比率计算口径
  - [x] 3.1: 在 `run_bootstrap_ci` 函数中，使用 `compute_backtest_metrics` 的sharpe_ratio作为点估计
  - [x] 3.2: 验证同一方法在不同实验中的夏普比率差异<0.01 ✅ B-LGBM: 0.7108一致

- [x] Task 4: 修复DSR计算异常
  - [x] 4.1: n_obs已正确使用len(returns)（126周）
  - [x] 4.2: 修复DSR函数：将年化夏普转换为非年化（/sqrt(52)），expected_max也做相同转换
  - [x] 4.3: 验证修复后DSR值在合理区间 ✅ 0.01~0.04（仍<0.95但不再是1e-79）

- [x] Task 5: 修复换手率全0问题
  - [x] 5.1: 在 `run_topn_sensitivity`、`run_holding_period_sensitivity`、`run_rrf_k_sensitivity` 中添加 `extended=True`
  - [x] 5.2: 验证修复后换手率 > 0 ✅ B-LGBM持有1周转手率=1.0295

- [x] Task 6: 修复持有期敏感性分析
  - [x] 6.1: 在 `BacktestEngine` 中添加 `rebalance_freq` 参数（默认1，单位周）
  - [x] 6.2: 修改 `run_backtest` 方法，当 `rebalance_freq > 1` 时，仅在调仓周执行选股，非调仓周保持持仓
  - [x] 6.3: 在 `run_statistical_tests.py` 的 `run_holding_period_sensitivity` 中使用 `rebalance_freq` 参数
  - [x] 6.4: 验证2周调仓和4周调仓的换手率低于1周调仓 ✅ 1.03 > 0.63 > 0.38

- [x] Task 7: 移除global变量
  - [x] 7.1: 将 `global n_trials` 改为 `generate_report` 的参数
  - [x] 7.2: 在 `main` 函数中将 `n_trials` 作为参数传入

- [x] Task 8: 重新运行统计检验和实验4
  - [x] 8.1: 重新运行 `python scripts/run_statistical_tests.py` ✅
  - [x] 8.2: 重新运行 `python scripts/run_experiment4.py` ✅
  - [x] 8.3: 验证所有修复点 ✅

# Task Dependencies
- Task 8 depends on Task 1, 2, 3, 4, 5, 6, 7
- Task 6.3 depends on Task 6.1, 6.2
