# 跑通基线实验 Spec

## Why
根据实验方法论文档，所有实验必须先跑通基线并确认基线结果合理。当前LightGBM模型严重欠拟合（best_iteration=8，测试集NDCG@10=0.323，年化收益-16.72%），XGBoost表现尚可但夏普比率低于0.5。需要重新训练模型、补充B-EW和B-MOM基线方法、运行完整基线回测、生成基线分析报告。

## What Changes
- 修复LightGBM训练参数（降低min_child_samples上限、增加learning_rate下限、扩大early_stopping轮数）
- 新增基线运行脚本 `scripts/run_baseline.py`，一键训练双模型+回测+生成报告
- 新增B-EW（等权随机选股）基线方法实现
- 新增B-MOM（纯动量排序选股）基线方法实现
- 生成基线实验报告 `data/baseline/baseline_report.md`
- **BREAKING**: LightGBM模型文件将被重新训练覆盖

## Impact
- Affected specs: 模型训练、回测引擎、基线实验
- Affected code: `src/model_lightgbm.py`（参数修复）、新增脚本、`models/lightgbm.pkl`（覆盖）

## ADDED Requirements

### Requirement: 基线运行脚本
系统 SHALL 提供一键运行基线实验的脚本，依次执行：训练LightGBM、训练XGBoost、双模型回测、B-EW回测、B-MOM回测、汇总对比。

#### Scenario: 运行基线脚本
- **WHEN** 用户运行 `python scripts/run_baseline.py`
- **THEN** 脚本依次执行所有基线实验，生成 `data/baseline/baseline_report.md` 和各基线的回测结果

### Requirement: B-EW等权随机选股基线
系统 SHALL 实现等权随机选股基线：从股票池中随机选N只等权配置，跑多次取平均。

#### Scenario: B-EW回测
- **WHEN** 运行B-EW基线
- **THEN** 从测试集每期截面随机选Top N只股票，等权配置，跑100次取平均净值和指标

### Requirement: B-MOM纯动量排序基线
系统 SHALL 实现纯动量排序选股基线：按6个月动量因子排序选股，无ML模型。

#### Scenario: B-MOM回测
- **WHEN** 运行B-MOM基线
- **THEN** 按mom_6m因子降序排列，选Top N只股票等权配置，执行回测

### Requirement: LightGBM训练参数修复
系统 SHALL 修复LightGBM训练参数，解决严重欠拟合问题。

#### Scenario: 修复后训练
- **WHEN** 使用修复后的参数训练LightGBM
- **THEN** best_iteration > 50，测试集NDCG@10 > 0.35，回测年化收益 > 0

### Requirement: 基线分析报告
系统 SHALL 生成Markdown格式的基线分析报告，包含所有基线方法的对比结果和合理性判断。

#### Scenario: 查看报告
- **WHEN** 用户打开 `data/baseline/baseline_report.md`
- **THEN** 报告包含：各基线指标对比表、净值曲线数据、合理性判断（是否在实验方法论规定的范围内）、问题诊断

## MODIFIED Requirements

### Requirement: LightGBM Optuna搜索空间
- `min_child_samples` 上限从100降低为50（避免过大导致欠拟合）
- `learning_rate` 下限从0.01提高为0.03（避免过小导致迭代不足）
- `EARLY_STOPPING_ROUNDS` 从20增加到50（给模型更多训练机会）

## REMOVED Requirements
无移除项。
