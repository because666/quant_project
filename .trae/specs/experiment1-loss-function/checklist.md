# Checklist

## 标签构造函数
- [x] `return_aware_relevance` 函数已实现（E1a：收益加权NDCG）
- [x] `sharpe_aware_relevance` 函数已实现（E1b：收益加权 + 夏普惩罚，sharpe_penalty=0.5）
- [x] `cvar_aware_relevance` 函数已实现（E1c：收益加权 + 夏普惩罚 + CVaR，cvar_penalty=0.3，cvar_alpha=0.10）
- [x] 三个函数的输出标签值域为 [0, max_label]
- [x] E1b和E1c标签差异已验证（平均差异0.23，最大差异11）

## 训练入口
- [x] `build_datasets_with_label_fn` 函数已实现
- [x] `train_final_lightgbm_with_label_fn` 函数已实现
- [x] 自定义标签训练可正常运行
- [x] 实验组early_stopping_rounds=150（基线保持50）
- [x] 学习率搜索范围0.01-0.1（基线保持0.03-0.2）

## 实验脚本
- [x] `scripts/run_experiment1.py` 已创建
- [x] 脚本可训练E1a/E1b/E1c三个模型
- [x] 脚本可运行回测并计算指标
- [x] `scripts/search_experiment1_params.py` 已创建

## 实验结果
- [x] E1a NDCG@10=0.2071 > B-LGBM NDCG@10=0.1629（排序质量提升）
- [x] E1b best_iteration=27, E1c best_iteration=32（均>8，修复生效）
- [x] E1b和E1c结果不同（年化差异4.23pp，夏普差异0.242）
- [x] E1b（vp=0.7）Calmar=1.238 > B-LGBM（vp=1.0）Calmar=1.011
- [x] E1c（vp=0.5）夏普=0.720 > B-LGBM（vp=1.0）夏普=0.711
- [x] 实验1对比报告已生成

## 基准净值
- [x] 基准净值获取增加重试机制（3次重试，5秒间隔）
- [x] 基准净值获取增加本地缓存（parquet格式）
- [x] 基准净值获取增加三级回退（akshare→baostock→本地数据）
- [x] 基准净值获取成功（100/100非空）
