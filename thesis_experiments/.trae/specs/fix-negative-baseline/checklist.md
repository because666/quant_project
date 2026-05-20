# Checklist

## 诊断验证
- [x] 模型分数与future_return_1w的Spearman相关性已计算（LightGBM=-0.0519, XGBoost=-0.0448）
- [x] 波动率暴露分析已完成
- [x] 过拟合程度已量化（LightGBM差距70%, XGBoost差距48%）
- [x] 合成close价格正确性已验证（相关系数=1.0）
- [x] 回测逻辑正确性已验证（现金占比=0%，成本拖累≈8%）

## 因子修复
- [x] factor_columns.pkl已移除6个全零因子
- [x] train/val/test.parquet已重新生成（34个因子列）
- [x] 新数据中因子列数量=34

## 模型正则化
- [x] LightGBM训练参数已增加正则化（min_child_samples=50, lambda_l1=0.5, lambda_l2=0.5, num_leaves≤63）
- [x] XGBoost训练参数已增加正则化（min_child_weight=10, alpha=0.5, lambda=2.0, max_depth≤8）
- [x] 训练-验证NDCG差距<50%（LightGBM: 0.12, XGBoost: 0.004）

## 波动率调整
- [x] BacktestEngine已添加vol_penalty参数
- [x] 选股时使用adjusted_score = model_score - vol_penalty * volatility_12w
- [x] 波动率调整后选股的波动率低于纯模型选股（回撤从62%降至16%）

## 模型重训练
- [x] LightGBM使用34因子+增强正则化重新训练成功
- [x] XGBoost使用34因子+增强正则化重新训练成功
- [x] 模型可正常加载和预测

## 基线验证
- [x] B-LGBM年化收益 > B-EW年化收益（+16.55% > -6.46% ✅）
- [x] B-XGB年化收益 > B-EW年化收益（+13.70% > -6.46% ✅）
- [x] 基线报告已生成（含诊断证据、修复前后对比）
