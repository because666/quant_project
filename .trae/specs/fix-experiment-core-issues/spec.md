# 修复实验核心问题 Spec

## Why
当前实验存在3个P0阻断问题：E1b/E1c标签函数导致模型严重欠训练（best_iteration=3）、实验4中XGBoost相关方法全部失败（回测结果全零）、实验4脚本中模型路径和FusionPredictor初始化逻辑有Bug。这些问题导致实验1结论可靠性存疑、实验4完全无效、论文数据与实际结果不一致。必须从根因修复，使实验结果可信、可复现。

## What Changes
- 修复E1b/E1c标签函数的标签区分度问题，使模型能正常训练（best_iteration > 20）
- 修复run_experiment4.py中XGBoost模型路径（.pkl → .json）和FusionPredictor初始化逻辑
- 重新训练E1b/E1c模型并验证best_iteration
- 重跑实验1（含vol_penalty网格搜索）和实验4（含统计检验）
- 清理旧的xgboost.pkl文件
- **BREAKING**: E1b/E1c模型文件将被重新训练覆盖，实验1/4结果将被更新

## Impact
- Affected specs: rebuild-baseline-experiment, fix-all-experiment-issues, consolidate-all-remaining-issues
- Affected code: thesis_experiments/src/model_lightgbm.py（标签函数）、thesis_experiments/scripts/run_experiment4.py（模型路径+FusionPredictor）
- Affected data: thesis_experiments/data/experiment1/、thesis_experiments/data/experiment4/ 下所有JSON结果
- Affected models: thesis_experiments/models/e1b_lightgbm.pkl、e1c_lightgbm.pkl

---

## 问题根因分析

### 根因1：E1b/E1c标签区分度极低导致best_iteration=3

**现象**：E1b和E1c的最新训练日志显示best_iteration=3，而之前训练曾达到best_iteration=27/40。

**训练日志对比**：
| 时间 | E1b best_iteration | E1b val_ndcg@10 | 备注 |
|------|-------------------|-----------------|------|
| 05-18 20:58 | 17 | 0.200839 | 旧版（54因子，含group_id） |
| 05-18 22:20 | 8 | 0.147776 | 52因子，early_stopping=50 |
| 05-18 23:07 | 27 | 0.155283 | 52因子，early_stopping=150 |
| 05-31 01:36 | 40 | 0.140147 | 52因子，early_stopping=150，Optuna搜索空间调整 |
| 05-31 10:47 | **3** | 0.147945 | 52因子，early_stopping=100，**当前版本** |

**关键发现**：best_iteration=3不是标签函数本身的问题，而是**Optuna搜索过程中early_stopping=100但最终训练时也用了early_stopping=100**，而Optuna搜索阶段用的是early_stopping=50。当Optuna找到的learning_rate=0.0596时，最终训练用early_stopping=150（代码第678行硬编码），但实际日志显示"early_stopping_rounds=150"时best_iteration=3。

**真正原因**：Optuna搜索阶段early_stopping=50，找到了适合50轮早停的超参（learning_rate=0.0596），但最终训练时用early_stopping=150，导致过拟合——模型在验证集上第3轮就达到最佳，之后持续过拟合训练集导致验证集NDCG下降150轮未改善。

**修复方案**：
1. 统一Optuna搜索和最终训练的early_stopping_rounds
2. 降低标签函数的惩罚力度，提高标签区分度
3. 增加Optuna搜索次数（20→50），让搜索更充分
4. 对E1b/E1c使用更小的learning_rate搜索范围

### 根因2：实验4 XGBoost模型路径错误

**现象**：run_experiment4.py第109行M2配置`"model_path": MODELS_DIR / "xgboost.pkl"`，但实际模型已保存为`xgboost.json`。

**修复方案**：将所有xgboost.pkl引用改为xgboost.json。

### 根因3：FusionPredictor初始化绕过正常流程

**现象**：run_experiment4.py第169行直接覆盖`fp._predictors`私有属性。

**修复方案**：重构create_predictor函数，直接传入子预测器列表而非绕过初始化。

---

## ADDED Requirements

### Requirement: E1b/E1c模型训练充分性
E1b和E1c模型训练后，best_iteration SHALL >= 20，验证集NDCG@10 SHALL > 0.10。

#### Scenario: 模型训练充分
- **WHEN** 使用修复后的标签函数和训练参数训练E1b/E1c模型
- **THEN** best_iteration >= 20，验证集NDCG@10 > 0.10

### Requirement: 实验4 XGBoost方法有效
实验4中M2-B-XGB、M3-RRF、M4-AVG SHALL 产生有效的非零回测结果。

#### Scenario: XGBoost回测有效
- **WHEN** 运行实验4
- **THEN** M2-B-XGB的annualized_return不为0，sharpe_ratio不为-1.37e+17

### Requirement: FusionPredictor正常初始化
FusionPredictor SHALL 通过公开接口初始化，不得直接覆盖私有属性。

#### Scenario: 融合预测器正常工作
- **WHEN** 创建FusionPredictor实例并传入子预测器
- **THEN** 融合预测器能正常产生非零预测分数

### Requirement: 模型路径一致性
所有脚本中的XGBoost模型路径 SHALL 使用.json格式，与model_xgboost.py的DEFAULT_MODEL_PATH一致。

#### Scenario: 模型路径正确
- **WHEN** 检查所有脚本中的XGBoost模型路径
- **THEN** 路径均指向xgboost.json而非xgboost.pkl

## MODIFIED Requirements

### Requirement: E1b标签函数
sharpe_aware_relevance SHALL 在保持标签区分度的前提下施加夏普惩罚，惩罚后标签值分布的标准差 SHALL >= 原始标签标准差的30%。

### Requirement: E1c标签函数
cvar_aware_relevance SHALL 在保持标签区分度的前提下施加CVaR惩罚，惩罚后非零标签比例 SHALL >= 50%。

### Requirement: Optuna搜索与最终训练一致性
Optuna搜索阶段的early_stopping_rounds SHALL 与最终训练阶段一致，均为EARLY_STOPPING_ROUNDS常量值。

## REMOVED Requirements
无移除项。
