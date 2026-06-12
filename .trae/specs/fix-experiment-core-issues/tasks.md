# Tasks

## 第一阶段：修复标签函数和训练逻辑

- [x] Task 1: 诊断E1b/E1c标签分布，确认标签区分度问题
  - [x] SubTask 1.1: 编写诊断脚本，输出E1b/E1c标签的统计信息
  - [x] SubTask 1.2: 对比E1a（正常）与E1b/E1c的标签分布差异
  - [x] SubTask 1.3: 确定标签函数修复方案（增大max_label+降低惩罚系数）

- [x] Task 2: 修复E1b标签函数sharpe_aware_relevance
  - [x] SubTask 2.1: sharpe_penalty默认值从0.5降至0.3
  - [x] SubTask 2.2: max_label默认值从30增至100
  - [x] SubTask 2.3: 同步修改project/backend/src/model_lightgbm.py

- [x] Task 3: 修复E1c标签函数cvar_aware_relevance
  - [x] SubTask 3.1: cvar_penalty默认值从1.5降至0.5
  - [x] SubTask 3.2: CVaR惩罚改为比例惩罚（cvar_penalty * distance * out[sl]）
  - [x] SubTask 3.3: 同步修改project/backend/src/model_lightgbm.py

- [x] Task 4: 修复train_final_lightgbm_with_label_fn的early_stopping不一致
  - [x] SubTask 4.1: 第678行硬编码early_stopping_rounds=150改为EARLY_STOPPING_ROUNDS常量
  - [x] SubTask 4.2: 同步修改project/backend/src/model_lightgbm.py

- [x] Task 5: 增加Optuna搜索次数和搜索空间优化
  - [x] SubTask 5.1: E1b/E1c的n_trials从20增加到50
  - [x] SubTask 5.2: 更新run_experiment1.py中的sharpe_penalty和cvar_penalty参数

## 第二阶段：修复实验4脚本

- [x] Task 6: 修复run_experiment4.py的XGBoost模型路径
  - [x] SubTask 6.1: 3处xgboost.pkl改为xgboost.json
  - [x] SubTask 6.2: 修复run_statistical_tests.py中4处xgboost.pkl引用
  - [x] SubTask 6.3: 修复model_evaluation.py和config.py中的xgboost.pkl引用

- [x] Task 7: 修复FusionPredictor初始化逻辑
  - [x] SubTask 7.1: FusionPredictor新增predictors参数
  - [x] SubTask 7.2: run_experiment4.py通过predictors参数传入子预测器
  - [x] SubTask 7.3: 同步修改project/backend/src/fusion.py

- [x] Task 8: 清理旧xgboost.pkl文件
  - [x] SubTask 8.1: 删除thesis_experiments/models/xgboost.pkl

## 第三阶段：重新训练模型和重跑实验

- [x] Task 9: 重新训练E1b模型（best_iteration=101, val_ndcg@10=0.5388）
- [x] Task 10: 重新训练E1c模型（best_iteration=17, val_ndcg@10=0.5415）
- [x] Task 11: 重新训练基线模型（B-LGBM best_iteration=450, B-XGB best_iteration=4）
- [x] Task 12: 重新训练E1a模型（best_iteration=43, val_ndcg@10=0.5138）
- [x] Task 13: 重跑实验1（skip-train模式）
- [x] Task 14: 重跑实验4和统计检验

## 第四阶段：额外修复

- [x] Task 15: 修复XGBoost ndcg_exp_gain限制（标签>31时需设置ndcg_exp_gain=False）
- [x] Task 16: 添加LightGBM label_gain参数（list(range(101))对应max_label=100）
- [x] Task 17: 同步所有修改到project/backend/src/

# Task Dependencies
所有任务已完成。
