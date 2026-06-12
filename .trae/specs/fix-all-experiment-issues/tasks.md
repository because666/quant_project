# Tasks

- [ ] Task 1: 修复E1a标签函数abs(y) Bug
  - [ ] SubTask 1.1: 修改 `thesis_experiments/src/model_lightgbm.py` 中 `return_aware_relevance` 函数，将 `abs_seg = np.abs(seg)` 改为 `pos_seg = np.maximum(seg, 0.0)`，后续 `bonus = np.power(pos_seg, alpha)`
  - [ ] SubTask 1.2: 同步修改 `project/backend/src/model_lightgbm.py`（共享文件副本）
  - [ ] SubTask 1.3: 编写单元测试验证修复：构造含负收益的截面数据，确认负收益股票bonus为0

- [ ] Task 2: 修复CVaR惩罚distance无上界
  - [ ] SubTask 2.1: 修改 `thesis_experiments/src/model_lightgbm.py` 中 `cvar_aware_relevance` 函数，将 `distance = np.maximum((threshold - seg) / abs(threshold), 0.0)` 改为 `distance = np.clip((threshold - seg) / abs(threshold), 0.0, 3.0)`
  - [ ] SubTask 2.2: 同步修改 `project/backend/src/model_lightgbm.py`
  - [ ] SubTask 2.3: 编写单元测试验证修复：构造极端负收益数据，确认distance不超过3.0

- [ ] Task 3: 修复vol_penalty与融合分数尺度耦合
  - [ ] SubTask 3.1: 在 `thesis_experiments/src/backtest.py` 的 `run_backtest` 方法中，对融合预测器（FusionPredictor）的分数做Min-Max归一化后再施加vol_penalty
  - [ ] SubTask 3.2: 同步修改 `project/backend/src/backtest.py`
  - [ ] SubTask 3.3: 编写单元测试验证：RRF融合分数归一化后范围在[0,1]

- [ ] Task 4: 统一实验1的vol_penalty对比
  - [ ] SubTask 4.1: 修改 `thesis_experiments/scripts/run_experiment1.py`，对B-LGBM/E1a/E1b/E1c分别搜索vol_penalty最优值（网格0.0, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0），使用各自最优值进行对比
  - [ ] SubTask 4.2: 在对比表中标注各模型使用的vol_penalty值

- [ ] Task 5: 重跑实验1（依赖Task 1, 2, 4）
  - [ ] SubTask 5.1: 删除旧模型文件 `thesis_experiments/models/e1a_lightgbm*.pkl`, `e1b_lightgbm*.pkl`, `e1c_lightgbm*.pkl`
  - [ ] SubTask 5.2: 运行 `python scripts/run_experiment1.py`，训练新模型并运行28组回测
  - [ ] SubTask 5.3: 检查新结果：E1a的NDCG应与B-LGBM有明显差异（之前因Bug导致几乎相同）

- [ ] Task 6: 重跑实验2（依赖Task 3, 5）
  - [ ] SubTask 6.1: 运行 `python scripts/run_experiment2.py`，使用修复后的回测引擎
  - [ ] SubTask 6.2: 检查新结果：RRF融合的夏普比率应有显著改善（之前0.287，预期>0.5）

- [ ] Task 7: 重跑实验4和统计检验（依赖Task 5, 6）
  - [ ] SubTask 7.1: 运行 `python scripts/run_experiment4.py`
  - [ ] SubTask 7.2: 运行 `python scripts/run_statistical_tests.py`
  - [ ] SubTask 7.3: 检查DSR检验结果是否改善

- [ ] Task 8: 更新论文数据（依赖Task 7）
  - [ ] SubTask 8.1: 将新实验结果更新到 `thesis/papers/paper_with_data.tex` 中所有表格
  - [ ] SubTask 8.2: 更新pgfplots图表数据

- [ ] Task 9: 修改论文3.3节LLM定位（独立任务）
  - [ ] SubTask 9.1: 将3.3节标题从"LLM可解释推荐效果验证"改为"面向用户的LLM投资推荐"
  - [ ] SubTask 9.2: 修改核心论点：LLM角色是"信息翻译器"，将专业指标翻译成普通用户能懂的自然语言建议
  - [ ] SubTask 9.3: 添加工程落地视角的论述：面向C端场景，技术指标对用户是噪音

- [ ] Task 10: 修改论文引言（独立任务）
  - [ ] SubTask 10.1: 添加深度学习等前沿技术段落，承认其进展
  - [ ] SubTask 10.2: 指出三大问题：过拟合（良文杯亲身经历）、训练成本高、可解释性差
  - [ ] SubTask 10.3: 引出LTR+树模型路线的合理性

- [ ] Task 11: 修正参考文献（独立任务）
  - [ ] SubTask 11.1: 搜索替换[6] Li et al. (2022)为真实存在的LTR选股论文
  - [ ] SubTask 11.2: 搜索替换[9] Zhang et al. (2020)为真实存在的多模型融合论文
  - [ ] SubTask 11.3: 更新references.bib和paper_with_data.tex中的引用

# Task Dependencies
- [Task 5] depends on [Task 1, Task 2, Task 4]
- [Task 6] depends on [Task 3, Task 5]
- [Task 7] depends on [Task 5, Task 6]
- [Task 8] depends on [Task 7]
- [Task 9] 独立，可与Task 1-8并行
- [Task 10] 独立，可与Task 1-8并行
- [Task 11] 独立，可与Task 1-8并行
