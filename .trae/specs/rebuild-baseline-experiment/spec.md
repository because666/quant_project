# 重建基线实验数据与重跑 Spec

## Why
当前基线实验存在多个根本性问题：group_id/group_size被错误纳入模型特征（LightGBM中group_id重要性排名第2，属于噪声拟合/数据泄露）、XGBoost模型文件与指标不一致（best_iteration 113 vs 55）、train/val/test.parquet文件已归档不在磁盘上、XGBoost基线回测年化-42%严重异常。这些问题导致基线实验结果不可信，必须从数据源头重新构建完整、可靠的数据集，修复模型训练流程，重新执行基线实验。

## What Changes
- 从归档或原始数据源重新获取完整的train/val/test.parquet数据集
- 从特征列表中移除group_id和group_size，仅保留52个真实因子
- 修复XGBoost模型保存逻辑，确保feature_names正确记录
- 重新训练LightGBM和XGBoost基线模型（使用正确的52因子特征集）
- 重新执行基线回测（B-LGBM、B-XGB、B-EW、B-MOM）
- 重新执行实验1-4和统计检验
- 生成完整的基线实验报告（含数据来源说明、实验步骤、结果分析、对比结论）
- **BREAKING**: 所有模型文件将被重新训练覆盖，所有实验结果将被更新

## Impact
- Affected specs: run-baseline, fix-negative-baseline, consolidate-all-remaining-issues
- Affected code: thesis_experiments/src/data_loader.py, thesis_experiments/src/model_lightgbm.py, thesis_experiments/src/model_xgboost.py, thesis_experiments/src/predictor.py, thesis_experiments/scripts/run_baseline.py
- Affected data: thesis_experiments/data/ 下所有parquet和JSON结果文件
- Affected models: thesis_experiments/models/ 下所有模型文件

---

## 问题分类与详情

### P0（阻断性问题，必须先修复）

#### P0-1：group_id/group_size作为模型特征
- **现状**: factor_columns.pkl和模型特征列表中包含group_id和group_size，这两个是排序学习的辅助列（截面编号和截面大小），不是因子
- **影响**: LightGBM中group_id的gain=792.61（排名第2），模型在利用截面编号作为预测信号，这是噪声拟合，严重影响泛化能力
- **修复**: 从特征列表中移除group_id和group_size，模型训练时仅使用52个真实因子

#### P0-2：train/val/test.parquet文件缺失
- **现状**: 数据已归档到project/backend/data_archives/的分卷压缩包中，磁盘上不存在
- **影响**: 无法直接运行模型训练和回测
- **修复**: 从归档解压或从原始数据源重新生成

#### P0-3：XGBoost模型文件不一致
- **现状**: xgboost.json中best_iteration=113（best_score=0.3419），但xgboost_metrics.json中记录best_iteration=55；feature_names和feature_types均为空数组
- **影响**: 模型文件可能不是最新训练的版本，预测时特征顺序可能错误
- **修复**: 重新训练XGBoost模型，确保保存时记录feature_names

### P1（重要问题，影响结果可信度）

#### P1-1：XGBoost基线回测极差
- **现状**: B-XGB年化-42.13%，夏普-1.666，远差于B-LGBM（7.12%/0.237）
- **可能原因**: max_depth=3过浅导致欠拟合、feature_names缺失导致特征顺序错误、group_id/group_size噪声
- **修复**: 修复特征问题后重新训练，调整参数搜索范围

#### P1-2：LightGBM best_iteration仅50
- **现状**: 训练仅50轮就触发早停，learning_rate=0.0125过低
- **影响**: 模型学习可能不充分
- **修复**: 提高learning_rate下限或增加early_stopping轮数

### P2（改进项，非阻断）

#### P2-1：数据覆盖率低的股票
- **现状**: 332只股票数据覆盖率<50%，因子值大量为NaN/填充值
- **影响**: 可能影响模型质量
- **修复**: 考虑在数据预处理阶段过滤低覆盖率股票

#### P2-2：6个全零因子残留检查
- **现状**: 之前修复移除了6个全零因子，但需确认factor_columns.pkl已更新
- **修复**: 验证factor_columns.pkl不包含全零因子

---

## ADDED Requirements

### Requirement: 数据集完整性验证
系统 SHALL 在生成train/val/test.parquet后，自动执行数据完整性校验，包括：行数、列数、时间范围、缺失值比例、全零列检测、group_id/group_size排除。

#### Scenario: 数据完整性校验通过
- **WHEN** 生成train/val/test.parquet后运行校验
- **THEN** 所有parquet文件行数>0、列数=52（仅真实因子）+辅助列、无全零因子列、group_id/group_size不在特征列中

### Requirement: 特征列表排除辅助列
系统 SHALL 确保factor_columns.pkl和模型训练特征列表中不包含group_id和group_size，仅保留52个真实因子。

#### Scenario: 特征列表正确
- **WHEN** 加载factor_columns.pkl
- **THEN** 列表中不包含group_id和group_size，因子数量=52

### Requirement: XGBoost模型保存feature_names
系统 SHALL 在保存XGBoost模型时，确保feature_names和feature_types正确记录，预测时按特征名匹配。

#### Scenario: XGBoost特征名记录
- **WHEN** 保存XGBoost模型到文件
- **THEN** 模型文件中feature_names包含所有52个因子名，预测时按名称匹配特征

### Requirement: 基线实验可复现
系统 SHALL 提供一键运行的基线实验脚本，从数据获取到结果报告全流程可复现，记录所有关键参数和环境变量。

#### Scenario: 基线实验一键运行
- **WHEN** 运行基线实验脚本
- **THEN** 脚本自动完成数据获取→特征工程→模型训练→回测→报告生成，结果可复现

### Requirement: 基线实验报告
系统 SHALL 生成包含以下内容的完整基线实验报告：数据来源说明、数据集统计、实验步骤、模型训练参数、回测结果、与历史基线对比、统计检验结论。

#### Scenario: 报告完整性
- **WHEN** 打开基线实验报告
- **THEN** 报告包含数据来源、样本量、时间范围、因子列表、模型参数、NDCG指标、回测指标、Bootstrap CI、配对检验结果

## MODIFIED Requirements

### Requirement: 因子列配置
factor_columns.pkl SHALL 仅包含52个真实因子，排除group_id和group_size，排除6个全零因子（avg_turnover_4w/8w/12w、turnover_change_1w、high_low_range_4w、open_close_ratio_4w）。

### Requirement: 模型训练特征
LightGBM和XGBoost训练时 SHALL 使用factor_columns.pkl中的因子列表作为特征，不得包含group_id、group_size或任何辅助列。

### Requirement: XGBoost模型保存
XGBoost模型保存时 SHALL 使用save_model()方法并确保feature_names参数正确传入，同时生成配套的metrics JSON文件，best_iteration值必须一致。

## REMOVED Requirements
无移除项。
