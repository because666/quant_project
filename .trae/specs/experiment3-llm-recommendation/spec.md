# 实验3：LLM可解释推荐评估 Spec

## Why
实验方法论文档4.4节定义了实验3——论文创新点②的LLM可解释推荐部分。LLM推荐的价值在于"可解释性"而非"收益提升"，底层信号相同收益自然接近。实验3需要展示不同信息粒度下LLM推荐的质量差异，证明特征贡献分解+LLM推荐能提供更有价值的投资建议。

## What Changes
- 新增 `src/feature_contribution.py` 模块，实现SHAP近似特征贡献分解
- 新增 `scripts/run_experiment3.py` 脚本，运行实验3并生成案例报告
- 生成E3a/E3b/E3c三种信息粒度下的LLM推荐案例

## Impact
- Affected code: 新增 `src/feature_contribution.py`
- Affected files: 新增 `data/experiment3/` 目录下的案例报告

## ADDED Requirements

### Requirement: SHAP近似特征贡献分解
系统 SHALL 提供 `compute_feature_contribution` 函数，使用LightGBM内置的特征贡献值（predict_contrib）计算每只股票的Top因子贡献分解。

#### Scenario: 特征贡献计算
- **WHEN** 调用 `compute_feature_contribution(model, panel_df, top_k=5)`
- **THEN** 返回每只股票的Top-K因子名称和贡献值

### Requirement: 三种信息粒度的提示词构造
系统 SHALL 提供三种不同信息粒度的提示词构造函数：
- E3a：模型输出 + Top因子名称（仅特征重要性排名）
- E3b：模型输出 + Top因子名称 + 特征贡献分解（每只股票的Top因子贡献值）
- E3c：模型输出 + Top因子名称 + 特征贡献分解 + LLM推荐（完整链路）

#### Scenario: E3a提示词
- **WHEN** 构造E3a提示词
- **THEN** 包含Top-K股票代码+得分+全局Top5因子名称

#### Scenario: E3b提示词
- **WHEN** 构造E3b提示词
- **THEN** 在E3a基础上增加每只股票的Top3因子贡献分解

#### Scenario: E3c提示词
- **WHEN** 构造E3c提示词
- **THEN** 在E3b基础上调用DeepSeek生成推荐建议

### Requirement: 实验3运行脚本
系统 SHALL 提供 `scripts/run_experiment3.py` 脚本，选取2个典型截面（如牛市周、熊市周），生成E3a/E3b/E3c三种信息粒度下的LLM推荐案例，输出对比报告。

#### Scenario: 运行实验3
- **WHEN** 运行 `python scripts/run_experiment3.py`
- **THEN** 生成2个截面的3种推荐案例，保存到 `data/experiment3/`

### Requirement: 一致性评估
系统 SHALL 计算LLM建议操作与模型排序信号的一致率。

#### Scenario: 一致性计算
- **WHEN** LLM建议买入某股票
- **THEN** 检查该股票是否在模型Top-K中，计算一致率

## MODIFIED Requirements
无修改。

## REMOVED Requirements
无移除。
