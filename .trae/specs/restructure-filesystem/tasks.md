# Tasks

- [ ] Task 1: 创建三级目录结构
  - [ ] 1.1: 创建 `project/`、`thesis/`、`thesis_experiments/` 顶级目录
  - [ ] 1.2: 在各顶级目录下创建子目录结构

- [ ] Task 2: 迁移 project/ 文件
  - [ ] 2.1: 迁移 `backend/` → `project/backend/`（src、tests、配置文件等）
  - [ ] 2.2: 迁移 `frontend/` → `project/frontend/`
  - [ ] 2.3: 迁移根目录配置文件（.gitignore、README.md、Procfile等）→ `project/`
  - [ ] 2.4: 迁移 `.trae/rules/`、`.trae/skills/` → `project/.trae/`
  - [ ] 2.5: 迁移项目运行数据（data/meta/、data/query_smoke_out/、data_archives/）→ `project/backend/data/`
  - [ ] 2.6: 迁移项目工具脚本和根目录散落文件
  - [ ] 2.7: 迁移 `docs/开发规范手册.md` → `project/docs/`

- [ ] Task 3: 迁移 thesis/ 文件
  - [ ] 3.1: 迁移论文写作指南、协作方案、行动计划 → `thesis/docs/`
  - [ ] 3.2: 迁移实验方法论文档 → `thesis/docs/`
  - [ ] 3.3: 迁移参考文献 → `thesis/docs/`
  - [ ] 3.4: 迁移组会PPT、演讲稿、PPT截图、Mermaid图表 → `thesis/ppt/`

- [ ] Task 4: 迁移 thesis_experiments/ 文件
  - [ ] 4.1: 迁移实验脚本 → `thesis_experiments/scripts/`
  - [ ] 4.2: 迁移实验数据/结果 → `thesis_experiments/data/`
  - [ ] 4.3: 迁移论文图表 → `thesis_experiments/figures/`
  - [ ] 4.4: 迁移模型文件 → `thesis_experiments/models/`
  - [ ] 4.5: 迁移实验相关测试 → `thesis_experiments/tests/`
  - [ ] 4.6: 迁移 `.trae/specs/` → `thesis_experiments/.trae/specs/`
  - [ ] 4.7: 迁移 `example/qlib/` → `thesis_experiments/reference/qlib/`

- [ ] Task 5: 处理共享文件
  - [ ] 5.1: 在 `thesis_experiments/` 中创建共享模块的副本（model_lightgbm.py、backtest.py等）
  - [ ] 5.2: 为副本添加注释说明双重用途及关联关系
  - [ ] 5.3: 确保 `project/` 中的原始文件保持完整

- [ ] Task 6: 修复路径引用
  - [ ] 6.1: 修复 `project/backend/` 中所有import路径
  - [ ] 6.2: 修复 `thesis_experiments/scripts/` 中所有数据路径和sys.path
  - [ ] 6.3: 修复前端API调用路径
  - [ ] 6.4: 更新 `.trae/rules/project_rules.md` 目录结构说明
  - [ ] 6.5: 更新 `.gitignore` 路径规则

- [ ] Task 7: 质量验证
  - [ ] 7.1: 验证 `project/` 目录后端可正常启动
  - [ ] 7.2: 验证 `thesis_experiments/` 实验脚本可正常运行
  - [ ] 7.3: 验证 `thesis/` 文件可正常访问
  - [ ] 7.4: 验证共享文件副本内容一致性

- [ ] Task 8: 版本控制提交
  - [ ] 8.1: 提交所有文件变更，添加清晰的变更说明
  - [ ] 8.2: 推送到远程仓库

# Task Dependencies
- Task 1 是所有迁移任务的前置条件
- Task 2-4 可并行执行
- Task 5 依赖 Task 2 和 Task 4
- Task 6 依赖 Task 2-5
- Task 7 依赖 Task 6
- Task 8 依赖 Task 7
