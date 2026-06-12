# Tasks

## 第一阶段：数据重建与验证

- [x] Task 1: 恢复train/val/test.parquet数据文件
  - [x] SubTask 1.1: 确认parquet文件存在于磁盘（train:766,196行/57列, val:232,150行, test:225,566行/63列）
  - [x] SubTask 1.2: 验证时间范围（train:2014-01~2020-12, val:2021-01~2022-12, test:2023-01~2024-12）

- [x] Task 2: 修复特征列表，排除group_id/group_size
  - [x] SubTask 2.1: 确认factor_columns.pkl包含group_id/group_size（54因子→52因子）
  - [x] SubTask 2.2: 修改data_loader.py，添加NON_FACTOR_COLUMNS和ZERO_FACTOR_COLUMNS常量
  - [x] SubTask 2.3: 重新生成factor_columns.pkl（52个真实因子）
  - [x] SubTask 2.4: 同步修改project/backend/src/data_loader.py

- [x] Task 3: 数据完整性校验
  - [x] SubTask 3.1: 验证52个因子列无全零、无异常缺失
  - [x] SubTask 3.2: 验证train/val/test时间切分正确

## 第二阶段：模型训练流程修复

- [x] Task 4: 修复LightGBM训练流程
  - [x] SubTask 4.1: 添加_EXCLUDE_COLS常量，训练时过滤group_id/group_size
  - [x] SubTask 4.2: learning_rate下限0.01→0.03，early_stopping 50→100
  - [x] SubTask 4.3: 同步修改project/backend/src/model_lightgbm.py

- [x] Task 5: 修复XGBoost训练和保存流程
  - [x] SubTask 5.1: 添加_EXCLUDE_COLS常量，训练时过滤group_id/group_size
  - [x] SubTask 5.2: 默认保存路径从xgboost.pkl改为xgboost.json
  - [x] SubTask 5.3: 修复predictor.py中XGBoost默认路径
  - [x] SubTask 5.4: 同步修改project/backend/src/model_xgboost.py和predictor.py

- [x] Task 6: 修复预测器特征匹配逻辑
  - [x] SubTask 6.1: predictor.py加载factor_columns后过滤group_id/group_size
  - [x] SubTask 6.2: 同步修改project/backend/src/predictor.py

## 第三阶段：重新训练基线模型

- [x] Task 7: 重新训练LightGBM基线模型
  - [x] SubTask 7.1: 使用52因子训练，best_iteration>50
  - [x] SubTask 7.2: B-LGBM无vp时年化-8.92%，夏普-0.300

- [x] Task 8: 重新训练XGBoost基线模型
  - [x] SubTask 8.1: 使用52因子训练，feature_names非空
  - [x] SubTask 8.2: B-XGB从-42%改善到-12.64%

## 第四阶段：重跑基线实验和全部实验

- [x] Task 9: 重跑基线实验
  - [x] SubTask 9.1: B-LGBM年化-8.92%, B-XGB年化-12.64%, B-EW年化-6.46%
  - [x] SubTask 9.2: B-XGB不再是-42%异常值
  - [x] SubTask 9.3: 修复run_baseline.py中NoneType格式化错误

- [x] Task 10: 重跑实验1
  - [x] SubTask 10.1: E1c夏普0.592 > B-LGBM夏普0.492（标签函数创新有效）
  - [x] SubTask 10.2: B-LGBM(vp=5.0): 年化13.80%, 夏普0.492

- [x] Task 11: 重跑实验2
  - [x] SubTask 11.1: B-XGB从-42%改善到-14.29%
  - [x] SubTask 11.2: Stacking(E2c)夏普0.223，唯一正收益融合策略

- [x] Task 12: 重跑实验4和统计检验
  - [x] SubTask 12.1: 实验4完成
  - [x] SubTask 12.2: 统计检验完成，0/15组显著（样本量不足）
  - [x] SubTask 12.3: 配对检验功效不足，需更长时间窗口

## 第五阶段：生成完整实验报告

- [x] Task 13: 生成完整基线实验报告
  - [x] SubTask 13.1: 报告包含数据来源说明
  - [x] SubTask 13.2: 报告包含数据集统计
  - [x] SubTask 13.3: 报告包含实验步骤和模型训练参数
  - [x] SubTask 13.4: 报告包含回测结果和与历史基线对比
  - [x] SubTask 13.5: 报告包含统计检验结论

- [x] Task 14: 更新论文实验数据
  - [x] SubTask 14.1: paper_with_data.tex中17处数据更新
  - [x] SubTask 14.2: 所有表格和图表数据已更新
  - [x] SubTask 14.3: 结论部分已更新

# Task Dependencies
- [Task 2] depends on [Task 1]
- [Task 3] depends on [Task 2]
- [Task 4] depends on [Task 2]
- [Task 5] depends on [Task 2]
- [Task 6] depends on [Task 2]
- [Task 7] depends on [Task 2, Task 4]
- [Task 8] depends on [Task 5, Task 7]
- [Task 9] depends on [Task 7, Task 8]
- [Task 10] depends on [Task 9]
- [Task 11] depends on [Task 10]
- [Task 12] depends on [Task 10, Task 11]
- [Task 13] depends on [Task 9]
- [Task 14] depends on [Task 12]
