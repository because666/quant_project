# Tasks

- [x] Task 1: 实现融合模块 `src/fusion.py`
  - [x] SubTask 1.1: 实现 `score_average_fusion` 函数（E2a）
  - [x] SubTask 1.2: 实现 `reciprocal_rank_fusion` 函数（E2b，返回排名+RRF分数）
  - [x] SubTask 1.3: 实现 `stacking_fusion` 函数（E2c，Ridge元学习器）
  - [x] SubTask 1.4: 实现 `weighted_rrf_fusion` 函数（E2d，NDCG加权）
  - [x] SubTask 1.5: 实现 `FusionPredictor` 类

- [x] Task 2: 修改回测引擎支持融合预测器
  - [x] SubTask 2.1: 在 `BacktestEngine` 中支持传入 `custom_predictor` 参数
  - [x] SubTask 2.2: 验证融合预测器可正常运行回测

- [x] Task 3: 实现实验2运行脚本
  - [x] SubTask 3.1: 创建 `scripts/run_experiment2.py`
  - [x] SubTask 3.2: 运行各融合策略回测（含波动率调整和参数搜索）
  - [x] SubTask 3.3: 生成实验2对比报告

- [x] Task 4: 修复RRF融合分数转换问题
  - [x] SubTask 4.1: 修改 `reciprocal_rank_fusion` 返回RRF分数字典
  - [x] SubTask 4.2: 修改 `FusionPredictor.predict` 使用RRF分数而非排名转分数
  - [x] SubTask 4.3: 重新运行实验2验证修复效果

- [x] Task 5: 运行实验2并验证结果
  - [x] SubTask 5.1: E2b-RRF换手率0.61 < B-LGBM换手率1.03 ✅
  - [x] SubTask 5.2: E2a分数平均年化16.32% ≈ B-LGBM 16.55%，夏普0.727 > 0.711 ✅
  - [x] SubTask 5.3: 实验2对比报告已生成

# Task Dependencies
- Task 1 → Task 2 → Task 3 → Task 4 → Task 5
