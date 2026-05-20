# 文件系统重构 Spec

## Why
项目文件混合存放，开发代码、学术论文文档和实验数据交织在一起，不利于协作分工和独立访问。需要建立清晰的目录结构，将项目开发、论文写作、论文实验三类文件明确分离。

## What Changes
- 建立三个顶级目录：`project/`、`thesis/`、`thesis_experiments/`
- 将现有文件按分类标准迁移至对应目录
- 处理共享文件（项目运行+实验共用），在对应目录创建副本
- 修复因文件移动导致的路径引用错误
- 更新版本控制和项目文档

### 文件分类标准

**project/** — 项目开发文件：
- 前后端源代码（`backend/src/`、`frontend/`）
- 部署配置（Dockerfile、Procfile、nginx.conf等）
- 项目运行数据（股票池元数据、冒烟测试数据、分卷数据包）
- 开发规范文档
- 工具脚本（数据下载、数据库初始化等）
- 项目功能测试
- `.trae/rules/`、`.trae/skills/`
- 根目录配置文件（.gitignore、README.md等）

**thesis/** — 学术论文文件：
- 论文写作指南、协作方案、行动计划
- 实验方法论文档
- 参考文献
- 组会PPT、演讲稿、PPT截图、Mermaid图表源码

**thesis_experiments/** — 论文实验相关文件：
- 实验脚本（run_experiment*.py、run_statistical_tests.py、generate_figures.py等）
- 实验数据/结果（experiment1~4/、baseline/、backtest_results/、statistical_tests/）
- 论文图表（figures/）
- 模型文件（models/）
- 实验相关测试（test_experiment*.py、test_statistical_tests.py）
- `.trae/specs/`（实验相关规格文档）
- Qlib参考代码库（example/qlib/）

### 共享文件处理
以下文件同时用于项目运行和论文实验，需在两个目录中各保留一份：
- `backend/src/model_lightgbm.py` → project + thesis_experiments
- `backend/src/model_xgboost.py` → project + thesis_experiments
- `backend/src/backtest.py` → project + thesis_experiments
- `backend/src/metrics.py` → project + thesis_experiments
- `backend/src/feature_engineering.py` → project + thesis_experiments
- `backend/src/fusion.py` → project + thesis_experiments
- `backend/src/feature_contribution.py` → project + thesis_experiments
- `backend/src/predictor.py` → project + thesis_experiments
- `backend/src/model_evaluation.py` → project + thesis_experiments
- `backend/src/data_loader.py` → project + thesis_experiments

## Impact
- Affected code: 所有Python脚本中的import路径、前端API调用路径
- **BREAKING**: 目录结构完全改变，所有相对路径引用需要更新
- `.trae/rules/project_rules.md` 需要更新目录结构说明

## ADDED Requirements

### Requirement: 三级目录结构
系统 SHALL 提供三个独立的顶级目录 `project/`、`thesis/`、`thesis_experiments/`，分别存放项目开发文件、学术论文文件和论文实验相关文件。

#### Scenario: 项目从project目录独立运行
- **WHEN** 用户在 `project/` 目录下执行后端启动命令
- **THEN** 后端服务正常启动，API可正常访问

#### Scenario: 实验从thesis_experiments目录独立运行
- **WHEN** 用户在 `thesis_experiments/` 目录下执行实验脚本
- **THEN** 实验脚本正常运行，结果输出到正确位置

#### Scenario: 论文文件独立访问
- **WHEN** 用户浏览 `thesis/` 目录
- **THEN** 所有论文相关文档、PPT、图表均可正常访问

## MODIFIED Requirements

### Requirement: 路径引用
所有Python脚本中的import路径和文件引用路径 SHALL 更新为新的目录结构。实验脚本中的 `sys.path` 和数据路径 SHALL 正确指向新位置。

## REMOVED Requirements
无
