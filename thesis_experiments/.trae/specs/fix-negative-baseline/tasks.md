# Tasks

- [x] Task 1: 深度诊断验证（已完成，结果已记录）
  - [x] SubTask 1.1: 编写并运行 `scripts/diagnose_baseline.py`
  - [x] SubTask 1.2: 编写并运行 `scripts/verify_backtest_detail.py`
  - [x] SubTask 1.3: 确认根因：模型分数与收益负相关 + 严重过拟合 + 高波动暴露

- [x] Task 2: 移除6个无效因子并重新生成数据
  - [x] SubTask 2.1: 修改 `data/factor_columns.pkl`，移除6个全为0的因子
  - [x] SubTask 2.2: 重新生成 train/val/test.parquet（不含无效因子列）
  - [x] SubTask 2.3: 验证新数据中因子列数量为34

- [x] Task 3: 增强模型正则化
  - [x] SubTask 3.1: 修改 `src/model_lightgbm.py`，增加正则化参数（min_child_samples=50, lambda_l1=0.5, lambda_l2=0.5, num_leaves上限=63）
  - [x] SubTask 3.2: 修改 `src/model_xgboost.py`，增加正则化参数（min_child_weight=10, alpha=0.5, lambda=2.0, max_depth上限=8）
  - [x] SubTask 3.3: 验证训练-验证NDCG差距降低（LightGBM: 0.37→0.12, XGBoost: 0.14→0.004）

- [x] Task 4: 添加波动率调整选股策略
  - [x] SubTask 4.1: 在 `BacktestEngine` 中添加 `vol_penalty` 参数
  - [x] SubTask 4.2: 在选股时使用 `adjusted_score = model_score - vol_penalty * volatility_12w`
  - [x] SubTask 4.3: 运行波动率调整选股回测，验证收益改善

- [x] Task 5: 重新训练模型并验证
  - [x] SubTask 5.1: 使用新的34因子数据+增强正则化重新训练LightGBM
  - [x] SubTask 5.2: 使用新的34因子数据+增强正则化重新训练XGBoost
  - [x] SubTask 5.3: 验证模型正常加载和预测

- [x] Task 6: 运行完整基线实验并生成报告
  - [x] SubTask 6.1: 运行 `scripts/run_baseline.py` 生成修复后的基线报告
  - [x] SubTask 6.2: 确认B-LGBM年化收益 > B-EW年化收益（+16.55% > -6.46% ✅）
  - [x] SubTask 6.3: 生成修复后的基线分析报告（含诊断证据、反转验证、波动率调整结果）

# Task Dependencies
- Task 2 → Task 5（需要新数据才能训练）
- Task 3 → Task 5（需要新参数才能训练）
- Task 4 → Task 6（需要波动率调整结果）
- Task 5 → Task 6（需要新模型才能跑基线）
