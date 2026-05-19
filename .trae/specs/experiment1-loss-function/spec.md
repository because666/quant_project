# 实验1：收益-夏普感知损失函数对比 Spec

## Why
实验方法论文档4.2节定义了实验1——论文最核心的创新实验。标准NDCG损失优化排序质量，但投资收益取决于Top-K股票的绝对收益水平。收益-夏普感知损失直接优化投资目标，预期NDCG略降但夏普比率显著提升。这是论文最有说服力的发现。

## What Changes
- 在 `model_lightgbm.py` 中新增 `return_aware_relevance` 标签构造函数（E1a：收益加权NDCG）
- 在 `model_lightgbm.py` 中新增 `sharpe_aware_relevance` 标签构造函数（E1b：收益加权 + 夏普惩罚）
- 在 `model_lightgbm.py` 中新增 `cvar_aware_relevance` 标签构造函数（E1c：收益加权 + 夏普惩罚 + CVaR约束）
- 新增 `build_datasets_with_label_fn` 函数，支持自定义标签构造函数
- 新增 `train_experiment1` 入口函数，训练E1a/E1b/E1c三个模型
- 新增 `scripts/run_experiment1.py` 脚本，运行实验1并生成对比报告

## Impact
- Affected code: `src/model_lightgbm.py`（新增标签构造函数和训练入口）
- Affected files: 新增 `scripts/run_experiment1.py`，新增 `models/e1a_lightgbm*.pkl` 等模型文件

## ADDED Requirements

### Requirement: E1a 收益加权NDCG标签构造
系统 SHALL 提供 `return_aware_relevance` 函数，在标准排名标签基础上，根据实际收益率大小额外提升高收益股票的标签值。

#### Scenario: E1a标签构造
- **WHEN** 调用 `return_aware_relevance(y, group_sizes, max_label=30, alpha=1.0)`
- **THEN** 高收益股票的relevance标签高于纯排名标签，alpha控制收益敏感度

### Requirement: E1b 收益加权 + 夏普惩罚标签构造
系统 SHALL 提供 `sharpe_aware_relevance` 函数，在E1a基础上，对Top-K组合中高波动股票的标签进行惩罚，降低其被选中的概率。

#### Scenario: E1b标签构造
- **WHEN** 调用 `sharpe_aware_relevance(y, group_sizes, volatility, max_label=30, alpha=1.0, sharpe_penalty=0.1)`
- **THEN** 高收益但高波动的股票标签被惩罚性降低

### Requirement: E1c 收益加权 + 夏普惩罚 + CVaR约束标签构造
系统 SHALL 提供 `cvar_aware_relevance` 函数，在E1b基础上，对左尾收益（极端亏损）的股票额外惩罚。

#### Scenario: E1c标签构造
- **WHEN** 调用 `cvar_aware_relevance(y, group_sizes, volatility, max_label=30, alpha=1.0, sharpe_penalty=0.1, cvar_penalty=0.05)`
- **THEN** 左尾收益极端的股票标签被进一步惩罚

### Requirement: 自定义标签训练入口
系统 SHALL 提供 `train_final_lightgbm_with_label_fn` 函数，支持传入自定义标签构造函数进行训练。

#### Scenario: 使用自定义标签训练
- **WHEN** 调用 `train_final_lightgbm_with_label_fn(label_fn=return_aware_relevance, ...)`
- **THEN** 模型使用指定的标签构造函数训练，模型保存到指定路径

### Requirement: 实验1运行脚本
系统 SHALL 提供 `scripts/run_experiment1.py` 脚本，一键运行E1a/E1b/E1c训练和回测，生成对比报告。

#### Scenario: 运行实验1
- **WHEN** 运行 `python scripts/run_experiment1.py`
- **THEN** 输出B-LGBM/E1a/E1b/E1c的NDCG@10、年化收益、夏普比率、最大回撤对比表

## MODIFIED Requirements
无修改。

## REMOVED Requirements
无移除。
