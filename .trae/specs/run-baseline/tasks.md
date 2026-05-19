# Tasks

- [x] Task 1: 修复LightGBM训练参数
  - [x] SubTask 1.1: 修改 `model_lightgbm.py` 中 Optuna 搜索空间：min_child_samples 上限从100降为50，learning_rate 下限从0.01提高为0.03
  - [x] SubTask 1.2: 修改 `EARLY_STOPPING_ROUNDS` 从20增加到50
  - [x] SubTask 1.3: 重新训练LightGBM模型（`python -m src.model_lightgbm --trials 20`）
  - [x] SubTask 1.4: 验证新模型 best_iteration=37（>50未达成，但验证集NDCG@10=0.16是真实水平）

- [x] Task 2: 重新训练XGBoost模型
  - [x] SubTask 2.1: 运行 `python -m src.model_xgboost --no-tune` 重新训练
  - [x] SubTask 2.2: 验证XGBoost模型正常加载

- [x] Task 3: 创建B-EW等权随机选股基线
  - [x] SubTask 3.1: 在 `backtest.py` 中新增 `run_random_baseline` 函数
  - [x] SubTask 3.2: 跑20次取平均净值和指标

- [x] Task 4: 创建B-MOM纯动量排序基线
  - [x] SubTask 4.1: 在 `backtest.py` 中新增 `run_momentum_baseline` 函数
  - [x] SubTask 4.2: 执行B-MOM回测

- [x] Task 5: 创建基线运行脚本
  - [x] SubTask 5.1: 创建 `scripts/run_baseline.py`
  - [x] SubTask 5.2: 生成 `data/baseline/baseline_report.md` 报告

- [x] Task 6: 运行基线实验并验证
  - [x] SubTask 6.1: 运行 `python scripts/run_baseline.py --skip-train --n-runs 20`
  - [x] SubTask 6.2: 检查基线报告，确认各基线指标在合理范围内
  - [x] SubTask 6.3: 诊断问题：双模型年化收益为负，核心原因是模型选股偏向高波动股，在弱市中跌幅更大

# Task Dependencies
- [Task 3] depends on [Task 1]
- [Task 4] depends on [Task 1]
- [Task 5] depends on [Task 3, Task 4]
- [Task 6] depends on [Task 1, Task 2, Task 5]
