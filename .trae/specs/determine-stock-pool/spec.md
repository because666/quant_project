# 确定最终股票池 Spec

## Why
当前项目使用2014-01-01前上市的2258只股票作为股票池，但项目规范要求"近10年存续A股约3000只"。对于论文实验，需要一个规模合理、有学术说服力的股票池，同时需要考虑数据可得性和计算效率。当前股票池偏少且筛选起始日期与"近10年"定义不一致。

## What Changes
- 新增股票池筛选脚本 `scripts/build_final_stock_pool.py`，综合多维度筛选条件确定最终股票池
- 生成最终股票池文件 `data/meta/final_stock_pool.parquet`，包含股票代码、名称、行业、上市日期、筛选标记
- 生成股票池分析报告 `data/meta/stock_pool_report.md`，包含筛选过程、规模对比、行业分布等统计信息
- 修改 `data_fetcher.py` 中 `get_stock_list` 的默认 `start_date` 参数，使其与"近10年"定义一致
- **BREAKING**: 默认股票池起始日期从 `2014-01-01` 调整为 `2016-01-01`，股票池规模从2258变为约2589只

## Impact
- Affected specs: 股票池筛选逻辑、数据获取模块
- Affected code: `src/data_fetcher.py`（默认参数修改）、`src/stock_pool.py`（无需修改，已有缓存）、新增脚本文件

## ADDED Requirements

### Requirement: 最终股票池筛选脚本
系统 SHALL 提供一个独立脚本，综合以下条件筛选最终股票池：

1. **存续期要求**：2016-01-01前上市且截至数据截止日未退市（"近10年存续"的合理定义）
2. **ST/退市标记**：排除名称含"退"字的股票，标记名称含"ST"的股票（不排除，仅标记）
3. **行业覆盖**：记录每只股票所属申万一级行业，确保行业分布合理
4. **数据完整性**：检查每只股票在周频因子数据中的覆盖率，标记覆盖率低于阈值的股票
5. **流动性筛选**：排除周均成交额低于一定阈值的股票（低流动性股票影响回测真实性）

#### Scenario: 运行筛选脚本
- **WHEN** 用户运行 `python scripts/build_final_stock_pool.py`
- **THEN** 脚本生成 `data/meta/final_stock_pool.parquet` 和 `data/meta/stock_pool_report.md`

#### Scenario: 股票池规模
- **WHEN** 筛选完成
- **THEN** 最终股票池规模约为2500-3000只，符合项目规范"约3000只"的要求

### Requirement: 股票池分析报告
系统 SHALL 生成Markdown格式的股票池分析报告，包含以下内容：

1. 筛选条件说明
2. 各筛选步骤的股票数量变化（漏斗图数据）
3. 最终股票池规模
4. 申万一级行业分布统计
5. 上市日期分布统计
6. 与现有股票池（2014起始2258只）的对比
7. 数据覆盖率统计

#### Scenario: 查看报告
- **WHEN** 用户打开 `data/meta/stock_pool_report.md`
- **THEN** 可清晰了解股票池筛选过程和最终结果

### Requirement: 默认股票池起始日期更新
系统 SHALL 将 `data_fetcher.py` 中 `get_stock_list` 的默认 `start_date` 从 `2014-01-01` 更新为 `2016-01-01`，与"近10年"定义一致（当前年份2026年，近10年=2016年起）。

#### Scenario: 默认调用
- **WHEN** 用户调用 `get_stock_list(strict_universe=True)` 不指定日期
- **THEN** 使用2016-01-01作为存续起始日期，返回约2589只股票

## MODIFIED Requirements

### Requirement: get_stock_list默认参数
`start_date` 参数默认值从 `"2014-01-01"` 修改为 `"2016-01-01"`。

## REMOVED Requirements
无移除项。
