# Checklist

## 融合模块
- [x] `score_average_fusion` 函数已实现（E2a）
- [x] `reciprocal_rank_fusion` 函数已实现（E2b，k=60，返回排名+RRF分数）
- [x] `stacking_fusion` 函数已实现（E2c，Ridge元学习器）
- [x] `weighted_rrf_fusion` 函数已实现（E2d，NDCG加权，返回排名+RRF分数）
- [x] `FusionPredictor` 类已实现，predict接口与ModelPredictor一致

## 回测集成
- [x] BacktestEngine支持传入custom_predictor参数
- [x] 融合预测器可正常运行回测
- [x] RRF分数修复：使用RRF原始分数而非排名转分数

## 实验脚本
- [x] `scripts/run_experiment2.py` 已创建
- [x] 脚本可运行E2a/E2b/E2c/E2d回测
- [x] 脚本可计算指标并生成报告

## 实验结果
- [x] E2a分数平均年化16.32% ≈ B-LGBM 16.55%，夏普0.727 > 0.711 ✅
- [x] E2b-RRF换手率0.61 < B-LGBM换手率1.03 ✅（RRF排名更稳定）
- [x] E2b-RRF年化7.59% < B-LGBM 16.55%（RRF排序区分度不足）
- [x] E2a胜率57.6% > B-LGBM 48.5% ✅
- [x] 实验2对比报告已生成
