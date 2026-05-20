# Tasks

- [x] Task 1: 实现特征贡献分解模块
  - [x] SubTask 1.1: 创建 `src/feature_contribution.py`，实现 `compute_feature_contribution` 函数
  - [x] SubTask 1.2: 验证特征贡献分解可正常运行

- [x] Task 2: 实现三种信息粒度的提示词构造
  - [x] SubTask 2.1: 实现E3a提示词（模型输出+Top因子名称）
  - [x] SubTask 2.2: 实现E3b提示词（+特征贡献分解）
  - [x] SubTask 2.3: 实现E3c提示词（+LLM推荐，网络不可用时降级跳过）

- [x] Task 3: 实现实验3运行脚本
  - [x] SubTask 3.1: 创建 `scripts/run_experiment3.py`
  - [x] SubTask 3.2: 选取2个典型截面（牛市周2024-09-20 + 熊市周2024-01-26）
  - [x] SubTask 3.3: 生成3种推荐案例并保存
  - [x] SubTask 3.4: 一致性评估（DeepSeek API不可用时跳过E3c）

- [x] Task 4: 运行实验3并验证结果
  - [x] SubTask 4.1: 运行 `python scripts/run_experiment3.py`
  - [x] SubTask 4.2: 验证E3a/E3b推荐案例已生成
  - [x] SubTask 4.3: 生成实验3对比报告

# Task Dependencies
- Task 1 → Task 2 → Task 3 → Task 4
