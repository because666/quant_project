# 综合问题修复 Spec

## Why
项目当前存在多个已修复但未验证的代码问题、部分修复的Bug、未完成的实验重跑、论文内容与实验结果不一致、以及代码同步缺失等问题。这些问题相互关联，需要系统性地梳理和修复，否则论文结论缺乏可信度，实验结果无法支撑论文声称。

## What Changes
- 修复vol_penalty归一化中volatility未归一化的量纲不匹配问题
- 重跑实验1-4和统计检验，获取修复后的完整实验结果
- 清理论文中的占位符和批注，用真实实验数据替换
- 修改论文3.3节LLM定位、引言深度学习论述、参考文献修正
- 同步backend/src/与thesis_experiments/src/的共享文件
- 更新fix-all-experiment-issues/tasks.md中已完成任务的状态

## Impact
- Affected specs: fix-all-experiment-issues, fix-core-issues, fix-negative-baseline
- Affected code: thesis_experiments/src/backtest.py, thesis_experiments/scripts/run_experiment*.py, thesis/papers/paper_with_data.tex, project/backend/src/

---

## 问题分类与详情

### A类：代码Bug（需修复或验证）

#### A1：E1a标签函数abs(y) Bug — 已修复，待验证
- **文件**: `thesis_experiments/src/model_lightgbm.py` 第111-112行
- **修复内容**: `np.abs(seg)` → `np.maximum(seg, 0.0)`，负收益股票bonus为0
- **同步状态**: `project/backend/src/model_lightgbm.py` 已同步修复
- **验证方式**: 构造含负收益的截面数据，确认负收益股票bonus为0
- **tasks.md状态**: 未更新（仍标记为`[ ]`）

#### A2：CVaR惩罚distance无上界 — 已修复，待验证
- **文件**: `thesis_experiments/src/model_lightgbm.py` 第217行
- **修复内容**: `np.maximum(...)` → `np.clip(..., 0.0, 3.0)`，distance上界截断为3.0
- **同步状态**: `project/backend/src/model_lightgbm.py` 已同步修复
- **验证方式**: 构造极端负收益数据，确认distance不超过3.0
- **tasks.md状态**: 未更新（仍标记为`[ ]`）

#### A3：vol_penalty归一化不完整 — 部分修复
- **文件**: `thesis_experiments/src/backtest.py` 第437-444行
- **当前状态**: score已做Min-Max归一化到[0,1]，但volatility_12w仍是原始值（典型范围0.1~0.6）
- **问题**: `vol_penalty * volatility_12w` 的惩罚量级与归一化后的score不匹配。当vol_penalty=0.5时，惩罚项约为0.05~0.30，而score范围为[0,1]，惩罚可能过大或过小
- **修复方案**: 对volatility_12w也做Min-Max归一化，使惩罚项与score在同一量纲
- **同步状态**: `project/backend/src/backtest.py` 已同步当前（部分修复）版本

#### A4：B-LGBM基线模型加载格式问题 — 已修复
- **文件**: `thesis_experiments/src/model_lightgbm.py` 第717-725行
- **修复内容**: `load_lightgbm_model` 函数使用 `lgb.Booster(model_file=...)` 加载，兼容LightGBM文本格式
- **验证方式**: 执行 `load_lightgbm_model()` 验证可正常加载
- **注意**: 修复后B-LGBM基线回测结果需重新验证（之前年化0.00%、夏普-1.37e+17的异常结果）

#### A5：backend/src/缺少数据库保护 — 未修复
- **文件**: `project/backend/src/backtest.py` 第38-39行
- **问题**: 直接 `from .db import init_db, session_scope`，无 `_HAS_DB` 保护
- **影响**: 在无数据库模块的环境中运行会直接报 `ImportError`
- **thesis_experiments版**: 已修复（try/except + `_HAS_DB` 标志）

### B类：实验问题（需重跑）

#### B1：实验1结果需重跑 — 部分完成
- **当前状态**: E1a/E1b/E1c新模型已训练完成，结果改善（E1b夏普0.772）
- **遗留问题**: B-LGBM基线回测结果异常（年化0.00%、夏普-1.37e+17），需用修复后的模型加载逻辑重跑
- **依赖**: A4（模型加载修复）已完成，可直接重跑

#### B2：实验2结果需重跑 — 未开始
- **依赖**: B1完成
- **预期**: RRF融合夏普比率应>0.5（修复前0.287）

#### B3：实验4和统计检验需重跑 — 未开始
- **依赖**: B1、B2完成
- **预期**: DSR检验结果应改善，配对检验至少2组显著

#### B4：实验结果与论文声称不一致 — 核心问题
- **论文声称**: "消融实验验证了三个创新组件的有效性"
- **实际结果**:
  - E1c夏普0.720 vs B-LGBM 0.711，差异不显著
  - RRF融合（E2b）夏普0.287远低于B-LGBM 0.711
  - DSR检验显示所有方法的夏普比率不可信
  - 仅1/15组配对检验显著
- **解决路径**: 重跑实验（B1-B3），用新结果更新论文

### C类：论文内容问题（需修改）

#### C1：论文占位符和批注未清理
- **文件**: `thesis/papers/paper_with_data.tex`
- **具体位置**:
  - 第63行: `【批注：摘要中占位数据已替换为真实实验结果。但需注意——当前实验结果显示收益-夏普感知损失函数未显著优于基线...】`
  - 第101-102行: `从2016年1月到2026年X月，涵盖约X000只A股上市公司` — 仍有X占位符
  - 第705-712行: `【批注：结论部分需根据真实实验结果重写——当前实验结果显示RRF融合未优于基线...】`

#### C2：论文3.3节LLM定位错误 — 未修改
- **当前标题**: "LLM可解释推荐效果验证"
- **应改为**: "面向用户的LLM投资推荐"
- **核心论点修改**: LLM角色是"信息翻译器"，将专业指标翻译成普通用户能懂的自然语言建议；面向C端场景，技术指标对用户是噪音

#### C3：论文引言表述基础 — 未修改
- **问题**: 只谈多因子模型显得落后，需承认深度学习等前沿技术
- **修改方向**: 添加深度学习前沿段落，指出三大问题（过拟合、训练成本高、可解释性差），引出LTR+树模型路线的合理性

#### C4：参考文献虚构引用 — 未修改
- **[6] Li et al. (2022)**: 原为虚构引用，已替换为Poh2021（需验证references.bib是否已更新）
- **[9] Zhang et al. (2020)**: 原为虚构引用，已替换为Alsulmi2022（需验证references.bib是否已更新）
- **需确认**: paper_with_data.tex中的引用是否与references.bib一致

### D类：代码同步与维护问题

#### D1：backend/src/与thesis_experiments/src/不同步
- **问题**: thesis_experiments版有最新修复（abs(y) Bug、CVaR截断、DB保护），backend版缺少
- **影响**: backend运行可能报错或使用旧逻辑
- **需同步文件**: backtest.py, model_lightgbm.py, fusion.py等

#### D2：fix-all-experiment-issues/tasks.md状态未更新
- **问题**: Task 1-2代码已修复但tasks.md仍标记为`[ ]`
- **影响**: 无法准确追踪任务进度

---

## ADDED Requirements

### Requirement: volatility归一化
系统 SHALL 在施加vol_penalty时，对volatility_12w也做Min-Max归一化，使惩罚项与归一化后的score在同一量纲[0,1]区间内。

#### Scenario: vol_penalty量纲匹配
- **WHEN** vol_penalty=0.5，volatility_12w原始值为0.3，score归一化后为0.6
- **THEN** 惩罚项 = 0.5 * (0.3 - vol_min) / (vol_max - vol_min)，惩罚项与score在同一量纲

### Requirement: 实验结果一致性
系统 SHALL 在所有代码修复完成后，按顺序重跑实验1→实验2→实验4→统计检验，确保所有实验结果基于同一套修复后的代码。

#### Scenario: 实验结果可复现
- **WHEN** 在修复后的代码环境中运行完整实验流程
- **THEN** 实验结果与论文声称一致，RRF融合优于单模型基线

### Requirement: 论文内容与实验结果一致
系统 SHALL 确保论文中所有数据、结论与实际实验结果一致，不得存在占位符、批注或与实验结果矛盾的声称。

#### Scenario: 论文无占位符
- **WHEN** 检查paper_with_data.tex
- **THEN** 不存在"X"占位符或"【批注】"标记

### Requirement: 共享文件双向同步
系统 SHALL 确保thesis_experiments/src/与project/backend/src/的共享文件内容完全一致。

#### Scenario: 文件同步验证
- **WHEN** 对比两个目录下的同名文件
- **THEN** 关键函数逻辑完全相同（允许导入路径差异）

## MODIFIED Requirements

### Requirement: 论文3.3节定位
3.3节 SHALL 从"LLM可解释推荐效果验证"改为"面向用户的LLM投资推荐"，核心论点改为"信息翻译器"定位。

### Requirement: 论文引言
引言 SHALL 承认深度学习前沿技术，指出过拟合/训练成本/可解释性三大问题，引出LTR+树模型路线的合理性。

### Requirement: 参考文献
参考文献[6]和[9] SHALL 替换为真实可查的论文，references.bib和paper_with_data.tex中的引用需一致。

## REMOVED Requirements
无移除项。
