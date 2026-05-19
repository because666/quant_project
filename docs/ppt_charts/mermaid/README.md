# PPT图表Mermaid源码汇总

本文档汇总了PPT中所有需要优化的图表的Mermaid源码，方便复制到Mermaid渲染工具中使用。

## 文件清单

| 页码 | 文件名 | 内容说明 | 优先级 |
|------|--------|----------|--------|
| 第4页 | slide04_a_share_comparison.md | A股vs美股交易制度对比 | 中 |
| 第6页 | slide06_tech_evolution.md | 五代技术演进路线图 | 高 |
| 第7页 | slide07_mse_vs_ranking.md | MSE vs 排序目标对比 | 🔴最高 |
| 第8页 | slide08_ndcg_explained.md | NDCG完整计算过程 | 高 |
| 第9页 | slide09_lambdarank_flow.md | LambdaRank原理流程 | 高 |
| 第10页 | slide10_sigmoid_explained.md | Sigmoid梯度项详解 | 高 |
| 第11页 | slide11_delta_ndcg.md | ΔNDCG代价计算 | 高 |
| 第12页 | slide12_lambdamart.md | LambdaMART训练流程 | 🔴最高 |
| 第14页 | slide14_backtest_results.md | 验证结果展示 | 高 |
| 第15页 | slide15_llm_integration.md | 大模型集成架构 | 🔴最高 |
| 第16页 | slide16_tech_choice.md | 技术选型对比 | 中 |
| 第17页 | slide17_team_strategy.md | 小团队生存策略 | 中 |
| 第18页 | slide18_compliance.md | 合规边界 | 中 |
| 第19页 | slide19_roadmap.md | 落地路径规划 | 中 |

## 使用方法

### 方法1: Mermaid Live Editor（推荐）
1. 访问 https://mermaid.live/
2. 复制对应文件中的Mermaid代码
3. 在线预览并导出为PNG/SVG/PDF

### 方法2: VS Code插件
1. 安装VS Code插件 "Markdown Preview Mermaid Support"
2. 在Markdown文件中使用 ```mermaid 代码块
3. 预览时自动渲染

### 方法3: Python转换
```python
# 使用mermaid-cli将Mermaid转为图片
# 需要先安装: npm install -g @mermaid-js/mermaid-cli

# 转换命令
mmdc -i slide07_mse_vs_ranking.md -o output.png -b white
```

## 图表类型说明

每个文件包含2-4个图表，覆盖以下类型：
- **流程图** (graph TD/LR): 展示算法流程、训练步骤
- **对比图** (subgraph): 对比不同方案、不同市场
- **架构图** (subgraph): 系统架构、模块关系
- **时间线** (graph LR): 技术演进、发展阶段

## 设计规范

所有图表遵循统一的颜色规范：
- 🟢 绿色 (#ccffcc): 正确/优势/可行
- 🔴 红色 (#ffcccc): 错误/劣势/禁止
- 🟠 橙色 (#fff3e0): 中间状态/警告
- 🔵 蓝色 (#e1f5ff): 信息/流程
- 🟣 紫色 (#f3e5f5): AI/大模型相关
- 深绿色 (#e8f5e9): 结论/重点

## 关键优化点

### 已解决的严重问题
1. **第7页**: 移除无关K线图，替换为MSE vs 排序目标对比图
2. **第12页**: 移除年龄预测图，替换为LambdaMART迭代修正排序图
3. **第15页**: 移除智能家居图，替换为大模型+量化系统架构图

### 新增的可视化
- 第9页: 新增LambdaRank原理流程图（原只有文字公式）
- 第10页: 新增Sigmoid三区标注图（原只有数学曲线）
- 第11页: 新增位置敏感性对比图（原只有文字描述）
