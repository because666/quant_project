# Tasks

## 第一阶段：代码Bug修复与验证

- [x] Task 1: 验证E1a标签函数abs(y) Bug修复
  - [x] SubTask 1.1: 编写单元测试，构造含负收益的截面数据，确认负收益股票bonus为0
  - [x] SubTask 1.2: 确认 `thesis_experiments/src/model_lightgbm.py` 第111-112行使用 `np.maximum(seg, 0.0)`
  - [x] SubTask 1.3: 确认 `project/backend/src/model_lightgbm.py` 已同步修复

- [x] Task 2: 验证CVaR惩罚distance截断修复
  - [x] SubTask 2.1: 编写单元测试，构造极端负收益数据，确认distance不超过3.0
  - [x] SubTask 2.2: 确认 `thesis_experiments/src/model_lightgbm.py` 第217行使用 `np.clip(..., 0.0, 3.0)`
  - [x] SubTask 2.3: 确认 `project/backend/src/model_lightgbm.py` 已同步修复

- [x] Task 3: 修复vol_penalty归一化中volatility未归一化问题
  - [x] SubTask 3.1: 修改 `thesis_experiments/src/backtest.py` 第437-444行，对volatility_12w也做Min-Max归一化
  - [x] SubTask 3.2: 同步修改 `project/backend/src/backtest.py`
  - [x] SubTask 3.3: 编写单元测试，验证惩罚项与score在同一量纲[0,1]区间

- [x] Task 4: 修复backend/src/缺少数据库保护问题
  - [x] SubTask 4.1: 修改 `project/backend/src/backtest.py`，添加try/except导入和 `_HAS_DB` 标志
  - [x] SubTask 4.2: 验证在无数据库模块的环境中运行不报错

- [x] Task 5: 同步backend/src/与thesis_experiments/src/的共享文件
  - [x] SubTask 5.1: 对比两个目录下的同名文件，列出差异
  - [x] SubTask 5.2: 将thesis_experiments版的最新修复同步到backend版
  - [x] SubTask 5.3: 验证同步后两端关键函数逻辑一致

## 第二阶段：重跑实验

- [x] Task 6: 重跑实验1（依赖Task 1, 2, 3）
  - [x] SubTask 6.1: 重新训练B-LGBM基线模型（54特征）和E1a/E1b/E1c模型
  - [x] SubTask 6.2: 运行 `python scripts/run_experiment1.py`，训练新模型并运行回测
  - [x] SubTask 6.3: 验证B-LGBM基线回测结果正常（年化7.12%，夏普0.237）
  - [x] SubTask 6.4: 验证E1a的NDCG与B-LGBM有明显差异（0.2166 vs 0.1594）

- [x] Task 7: 重跑实验2（依赖Task 6）
  - [x] SubTask 7.1: 运行 `python scripts/run_experiment2.py`
  - [x] SubTask 7.2: 融合结果不及预期（B-XGB拖累），E2c Stacking夏普0.078

- [x] Task 8: 重跑实验4和统计检验（依赖Task 6, 7）
  - [x] SubTask 8.1: 运行 `python scripts/run_experiment4.py`
  - [x] SubTask 8.2: 运行 `python scripts/run_statistical_tests.py`
  - [x] SubTask 8.3: 配对检验6/15组显著，DSR均不通过

## 第三阶段：论文内容修改

- [x] Task 9: 清理论文占位符和批注（依赖Task 8）
  - [x] SubTask 9.1: 替换"X"占位符为真实数据（2025年12月、约2500只）
  - [x] SubTask 9.2: 删除摘要区批注
  - [x] SubTask 9.3: 删除结论区批注，根据新实验结果重写结论
  - [x] SubTask 9.4: 全文搜索确认无残留占位符或批注

- [x] Task 10: 修改论文3.3节LLM定位
  - [x] SubTask 10.1: 标题已改为"面向用户的LLM投资推荐"
  - [x] SubTask 10.2: 核心论点改为"信息翻译器"定位
  - [x] SubTask 10.3: 添加工程落地视角论述

- [x] Task 11: 修改论文引言
  - [x] SubTask 11.1: 添加深度学习前沿技术段落
  - [x] SubTask 11.2: 指出三大问题：过拟合、训练成本高、可解释性差
  - [x] SubTask 11.3: 引出LTR+树模型路线的合理性

- [x] Task 12: 修正参考文献
  - [x] SubTask 12.1: references.bib中[6]已为Poh2021
  - [x] SubTask 12.2: references.bib中[9]已为Alsulmi2022
  - [x] SubTask 12.3: paper_with_data.tex引用与references.bib一致

- [x] Task 13: 更新论文实验数据（依赖Task 8, 9）
  - [x] SubTask 13.1: 将新实验结果更新到paper_with_data.tex中所有表格
  - [x] SubTask 13.2: 更新pgfplots图表数据
  - [x] SubTask 13.3: 论文结论已根据新结果重写

## 第四阶段：维护与清理

- [x] Task 14: 更新fix-all-experiment-issues/tasks.md状态
  - [x] SubTask 14.1: 将Task 1-2标记为已完成（代码已修复）
  - [x] SubTask 14.2: 将Task 3标记为已完成（归一化已修复）
  - [x] SubTask 14.3: 将Task 9-11标记为已完成（论文修改已完成）

# Task Dependencies
- [Task 6] depends on [Task 1, Task 2, Task 3]
- [Task 7] depends on [Task 6]
- [Task 8] depends on [Task 6, Task 7]
- [Task 9] depends on [Task 8]
- [Task 13] depends on [Task 8, Task 9]
- [Task 10] 独立，可与Task 1-8并行
- [Task 11] 独立，可与Task 1-8并行
- [Task 12] 独立，可与Task 1-8并行
- [Task 14] 在所有其他Task完成后执行
