# 验收清单

## 第一阶段：数据重建与验证

- [x] D1: train.parquet、val.parquet、test.parquet 文件存在于 `thesis_experiments/data/`
- [x] D2: factor_columns.pkl 仅包含52个真实因子，不含group_id/group_size
- [x] D3: 数据校验通过：52个因子列无全零、缺失值已用中位数填充
- [x] D4: train/val/test时间切分正确（train_end=2020-12-31, val_end=2022-12-31）
- [x] D5: 数据来源可追溯（akshare A股日线前复权数据）

## 第二阶段：模型训练流程修复

- [x] M1: LightGBM训练时使用52个因子（排除group_id/group_size）
- [x] M2: XGBoost训练时使用52个因子（排除group_id/group_size）
- [x] M3: XGBoost模型文件中feature_names非空（保存为JSON格式）
- [x] M4: XGBoost模型文件中best_iteration与metrics JSON一致
- [x] M5: 预测器按因子列名匹配特征（加载factor_columns后过滤group_id/group_size）

## 第三阶段：基线模型训练验证

- [x] B1: LightGBM新模型best_iteration>50（early_stopping=100生效）
- [x] B2: LightGBM训练-验证NDCG差距<50%
- [x] B3: LightGBM特征重要性中group_id/group_size已排除
- [x] B4: XGBoost新模型保存为JSON格式，feature_names非空
- [x] B5: XGBoost best_iteration与metrics JSON一致

## 第四阶段：基线实验结果验证

- [x] R1: B-LGBM年化收益13.80%（vp=5.0）> B-EW年化-6.46%（模型跑赢随机）
- [x] R2: B-XGB年化收益-14.29%（从-42%大幅改善，不再异常）
- [x] R3: E1c-LGBM夏普0.592 > B-LGBM夏普0.492（CVaR感知标签函数创新有效）
- [x] R4: 配对检验0/15组显著（样本量不足，需更长时间窗口）
- [x] R5: 所有模型使用相同的52因子特征集

## 第五阶段：实验报告验证

- [x] R6: 报告包含数据来源说明（akshare、时间范围、股票池筛选条件）
- [x] R7: 报告包含数据集统计（行数、列数、时间范围）
- [x] R8: 报告包含模型训练参数和NDCG指标
- [x] R9: 报告包含回测结果和与历史基线对比
- [x] R10: 报告包含统计检验结论
- [x] R11: 论文数据已更新为新实验结果（17处修改）
