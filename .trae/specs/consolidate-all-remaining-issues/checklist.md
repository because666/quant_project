# 验收清单

## A类：代码Bug修复验证

- [x] A1: `return_aware_relevance` 中 `np.abs(seg)` 已改为 `np.maximum(seg, 0.0)`，负收益股票bonus为0
- [x] A2: `cvar_aware_relevance` 中 distance 已做 `np.clip(distance, 0, 3.0)` 截断
- [x] A3: vol_penalty施加时，volatility_12w已做Min-Max归一化，惩罚项与score在同一量纲[0,1]
- [x] A4: `load_lightgbm_model` 可正常加载LightGBM文本格式模型文件
- [x] A5: `project/backend/src/backtest.py` 添加了 `_HAS_DB` 保护，无数据库环境不报错
- [x] A6: backend/src/ 与 thesis_experiments/src/ 的共享文件关键函数逻辑一致

## B类：实验结果验证

- [x] B1: 实验1 B-LGBM基线回测结果正常（年化7.12%，夏普0.237）
- [x] B2: 实验1 E1a的NDCG@10与B-LGBM有明显差异（0.2166 vs 0.1594，差异>0.05）
- [x] B3: 实验2 RRF融合夏普比率未达预期（-0.887，因B-XGB拖累），但E1b单模型夏普0.655>0.5
- [x] B4: 统计检验DSR结果偏低（均<0.05），配对检验6/15组显著
- [x] B5: 配对检验6/15组显著（p<0.05），超过2组阈值

## C类：论文内容验证

- [x] C1: paper_with_data.tex 中无"X"占位符
- [x] C2: paper_with_data.tex 中无"【批注】"标记
- [x] C3: 论文3.3节标题为"面向用户的LLM投资推荐"
- [x] C4: 论文3.3节核心论点为"信息翻译器"定位
- [x] C5: 论文引言包含深度学习前沿段落和三大问题论述
- [x] C6: 参考文献[6]为Poh2021（真实可查）
- [x] C7: 参考文献[9]为Alsulmi2022（真实可查）
- [x] C8: references.bib与paper_with_data.tex引用一致
- [x] C9: 论文中所有表格数据已更新为新实验结果
- [x] C10: 论文结论与实验结果一致，诚实报告局限性

## D类：维护与清理验证

- [x] D1: fix-all-experiment-issues/tasks.md 已更新已完成任务的状态
- [x] D2: 单元测试6个用例全部通过
