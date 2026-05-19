# 实验2：多模型排序融合对比 Spec

## Why
实验方法论文档4.3节定义了实验2——论文第二个创新点。单模型存在排序偏差，RRF融合利用多模型互补性提升排序稳定性。RRF不依赖分数绝对大小，仅使用排名位置，对模型输出尺度差异天然鲁棒，且排名稳定性更高可降低换手率。

## What Changes
- 新增 `src/fusion.py` 模块，实现4种融合策略（分数平均、RRF、Stacking、加权RRF）
- 新增 `scripts/run_experiment2.py` 脚本，运行实验2并生成对比报告
- 在 `BacktestEngine` 中支持传入融合预测器

## Impact
- Affected code: 新增 `src/fusion.py`，修改 `src/backtest.py`（支持融合预测器）
- Affected files: 新增模型文件、回测结果、对比报告

## ADDED Requirements

### Requirement: 分数平均融合（E2a）
系统 SHALL 提供 `score_average_fusion` 函数，将多个模型的预测分数归一化后取平均，生成融合排序。

#### Scenario: E2a融合
- **WHEN** 调用 `score_average_fusion(predictions_list, stock_codes)`
- **THEN** 返回融合后的分数DataFrame，按平均分数降序排列

### Requirement: RRF融合（E2b）
系统 SHALL 提供 `reciprocal_rank_fusion` 函数，实现Reciprocal Rank Fusion排序融合。

#### Scenario: E2b融合
- **WHEN** 调用 `reciprocal_rank_fusion(rankings, k=60, weights=None)`
- **THEN** 返回融合后的股票代码排序列表，RRF分数 = Σ w_i / (k + rank_i)

### Requirement: Stacking融合（E2c）
系统 SHALL 提供 `stacking_fusion` 函数，使用Ridge元学习器在验证集上学习最优融合权重。

#### Scenario: E2c融合
- **WHEN** 调用 `stacking_fusion(predictions_list, val_y, val_groups)`
- **THEN** 返回训练好的Ridge元学习器和融合预测函数

### Requirement: 加权RRF融合（E2d）
系统 SHALL 提供 `weighted_rrf_fusion` 函数，根据验证集NDCG@10为各模型分配权重。

#### Scenario: E2d融合
- **WHEN** 调用 `weighted_rrf_fusion(rankings, ndcg_weights, k=60)`
- **THEN** 返回融合后的排序列表，高NDCG模型权重更大

### Requirement: 融合预测器类
系统 SHALL 提供 `FusionPredictor` 类，封装融合逻辑，提供与 `ModelPredictor` 一致的 `predict` 接口。

#### Scenario: 融合预测器使用
- **WHEN** 创建 `FusionPredictor(fusion_type="rrf", model_types=["lightgbm", "xgboost"])`
- **THEN** 可调用 `predict(panel_df)` 获取融合后的分数

### Requirement: 实验2运行脚本
系统 SHALL 提供 `scripts/run_experiment2.py` 脚本，一键运行E2a/E2b/E2c/E2d回测，生成对比报告。

#### Scenario: 运行实验2
- **WHEN** 运行 `python scripts/run_experiment2.py`
- **THEN** 输出B-LGBM/B-XGB/E2a/E2b/E2c/E2d的年化收益、夏普比率、最大回撤、换手率对比表

## MODIFIED Requirements
无修改。

## REMOVED Requirements
无移除。
