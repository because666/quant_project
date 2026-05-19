# 修复基线负收益 Spec（更新版）

## Why
基线实验结果严重异常：全市场等权组合年化+9.43%，但模型Top20选股年化-23.69%（有成本）/ -15.70%（无成本）。经深度诊断确认：

1. **模型分数与未来收益负相关**（Spearman≈-0.05，75%截面呈负相关）
2. **严重过拟合**（LightGBM训练NDCG@10=0.54，验证=0.16，差距70%）
3. **高波动暴露导致严重波动率拖累**（算术平均周收益+0.74%，但复利年化-16.61%）
4. **6个全零因子**降低有效特征维度
5. **回测逻辑已验证正确**（合成close价格相关系数=1.0，现金占比=0%，成本拖累≈8%）

学术研究要求基线至少跑赢随机选股（B-EW），否则后续改进实验缺乏说服力。

## What Changes
- 移除6个全零因子（avg_turnover_4w/8w/12w、turnover_change_1w、high_low_range_4w、open_close_ratio_4w），从40→34个有效因子
- 增强模型正则化：增大min_child_samples、减小num_leaves/max_depth、增大lambda_l1/lambda_l2
- 添加波动率调整选股策略：adjusted_score = model_score - vol_penalty * volatility_12w
- 重新训练模型并验证基线至少跑赢B-EW
- 生成修复后的基线报告（含诊断证据）

## Impact
- Affected specs: 模型训练、因子体系、回测策略
- Affected code: `src/model_lightgbm.py`（增强正则化）、`src/model_xgboost.py`（增强正则化）、`src/backtest.py`（波动率调整选股）、因子列配置

## ADDED Requirements

### Requirement: 深度诊断验证
系统 SHALL 提供深度诊断脚本，验证模型分数与收益的相关性、波动率暴露、过拟合程度。

#### Scenario: 诊断脚本运行
- **WHEN** 运行 `scripts/diagnose_baseline.py`
- **THEN** 输出模型分数与future_return_1w的Spearman相关、波动率暴露比、过拟合程度等诊断信息

### Requirement: 移除无效因子
系统 SHALL 从因子列表中移除6个全为0的无效因子，避免降低模型有效特征维度。

#### Scenario: 移除后重新训练
- **WHEN** 移除无效因子后重新训练模型
- **THEN** 模型使用的因子数量从40减少到34，训练和预测正常

### Requirement: 增强正则化
系统 SHALL 在模型训练中增强正则化，防止过拟合导致测试集上分数与收益负相关。

#### Scenario: 正则化后过拟合减轻
- **WHEN** 使用增强正则化参数重新训练模型
- **THEN** 训练-验证NDCG差距从70%降低到50%以下

### Requirement: 波动率调整选股
系统 SHALL 在模型打分基础上添加波动率惩罚项，降低高波动股票的选中概率。

#### Scenario: 波动率调整选股回测
- **WHEN** 使用 `adjusted_score = model_score - vol_penalty * volatility_12w` 选股
- **THEN** 选出的股票波动率低于纯模型选股，年化收益优于纯模型选股

### Requirement: 基线跑赢B-EW
修复后的基线 SHALL 至少跑赢B-EW（等权随机选股）基线。

#### Scenario: 基线合理性
- **WHEN** 运行修复后的基线实验
- **THEN** B-LGBM年化收益 > B-EW年化收益（模型至少跑赢随机）

## MODIFIED Requirements

### Requirement: 因子列配置
`factor_columns.pkl` 中移除以下6个因子：avg_turnover_4w、avg_turnover_8w、avg_turnover_12w、turnover_change_1w、high_low_range_4w、open_close_ratio_4w

### Requirement: LightGBM训练参数
基础参数中增加正则化：
- min_child_samples: 从默认值增大到50-100
- lambda_l1: 增加L1正则化（0.1-1.0）
- lambda_l2: 增加L2正则化（0.1-1.0）
- num_leaves搜索范围上限从255降低到63

### Requirement: XGBoost训练参数
基础参数中增加正则化：
- min_child_weight: 从默认值增大到5-20
- alpha: 增加L1正则化（0.1-1.0）
- lambda: 增加L2正则化（1.0-5.0）
- max_depth搜索范围上限从10降低到8

## REMOVED Requirements
无移除项。
