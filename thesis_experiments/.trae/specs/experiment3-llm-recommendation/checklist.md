# Checklist

## 特征贡献分解
- [x] `compute_feature_contribution` 函数已实现
- [x] 使用LightGBM predict_contrib计算特征贡献值
- [x] 返回每只股票的Top-K因子名称和贡献值

## 提示词构造
- [x] E3a提示词已实现（模型输出+Top因子名称）
- [x] E3b提示词已实现（+特征贡献分解）
- [x] E3c提示词已实现（+LLM推荐，网络不可用时降级跳过）

## 实验脚本
- [x] `scripts/run_experiment3.py` 已创建
- [x] 选取2个典型截面（牛市周2024-09-20 + 熊市周2024-01-26）
- [x] 生成E3a/E3b推荐案例并保存到 data/experiment3/
- [x] 一致性评估逻辑已实现（DeepSeek API不可用时跳过）

## 实验结果
- [x] 2个截面的E3a/E3b推荐案例已生成
- [x] 3种信息粒度的对比报告已生成
- [x] E3c因DeepSeek API网络问题暂未生成（降级处理正常）
