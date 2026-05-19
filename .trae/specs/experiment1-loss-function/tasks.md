# Tasks

- [x] Task 1: 实现收益感知标签构造函数
  - [x] SubTask 1.1: 在 `model_lightgbm.py` 中新增 `return_aware_relevance` 函数（E1a）
  - [x] SubTask 1.2: 在 `model_lightgbm.py` 中新增 `sharpe_aware_relevance` 函数（E1b）
  - [x] SubTask 1.3: 在 `model_lightgbm.py` 中新增 `cvar_aware_relevance` 函数（E1c）

- [x] Task 2: 实现自定义标签训练入口
  - [x] SubTask 2.1: 在 `model_lightgbm.py` 中新增 `build_datasets_with_label_fn` 函数
  - [x] SubTask 2.2: 在 `model_lightgbm.py` 中新增 `train_final_lightgbm_with_label_fn` 函数
  - [x] SubTask 2.3: 验证自定义标签训练可正常运行

- [x] Task 3: 实现实验1运行脚本
  - [x] SubTask 3.1: 创建 `scripts/run_experiment1.py`，训练E1a/E1b/E1c模型
  - [x] SubTask 3.2: 在脚本中运行各模型回测（含波动率调整）
  - [x] SubTask 3.3: 生成实验1对比报告（NDCG@10、年化收益、夏普比率、最大回撤）

- [x] Task 4: 运行实验1并验证结果
  - [x] SubTask 4.1: 运行 `python scripts/run_experiment1.py`
  - [x] SubTask 4.2: 验证E1a年化收益 > B-LGBM年化收益
  - [x] SubTask 4.3: 验证E1b夏普比率 > B-LGBM夏普比率
  - [x] SubTask 4.4: 生成实验1对比报告

- [x] Task 5: 修复E1b/E1c结果一致问题
  - [x] SubTask 5.1: 增大sharpe_penalty从0.1到0.5，扩大惩罚范围到全截面
  - [x] SubTask 5.2: 增大cvar_penalty从0.05到0.3，cvar_alpha从0.05到0.10
  - [x] SubTask 5.3: 增大early_stopping_rounds从50到150（实验组）
  - [x] SubTask 5.4: 降低学习率搜索范围从0.03-0.2到0.01-0.1
  - [x] SubTask 5.5: 重新训练E1b/E1c，验证best_iteration>8且结果不同

- [x] Task 6: 修复基准净值获取问题
  - [x] SubTask 6.1: 增加重试机制（3次重试，5秒间隔）
  - [x] SubTask 6.2: 增加本地缓存（parquet格式）
  - [x] SubTask 6.3: 新增三级回退（akshare→baostock→本地数据）
  - [x] SubTask 6.4: 验证基准净值获取成功

# Task Dependencies
- Task 1 → Task 2 → Task 3 → Task 4 → Task 5 → Task 6
