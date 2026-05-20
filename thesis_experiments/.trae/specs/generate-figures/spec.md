# 整理实验结果与制作论文图表 Spec

## Why
4个实验和统计检验已完成，需要将JSON/MD格式的结果整理为论文可用的图表，包括分组柱状图、分位数单调性图、分年度对比图、敏感性分析图、Bootstrap CI误差棒图等，输出为PNG/SVG格式。

## What Changes
- 创建 `scripts/generate_figures.py` 脚本，读取所有实验结果JSON，生成论文图表
- 使用 matplotlib + seaborn 绘图，统一配色和字体
- 输出到 `data/figures/` 目录

## Impact
- Affected code: 新增脚本，不修改现有代码
- 依赖: matplotlib, seaborn（需确认已安装）

## ADDED Requirements

### Requirement: 论文图表生成脚本
系统 SHALL 提供 `scripts/generate_figures.py` 脚本，从实验结果JSON中读取数据并生成以下图表：

1. **图1：综合对比分组柱状图** — 4种方法的年化收益、夏普比率、最大回撤三指标分组对比
2. **图2：分位数单调性折线图** — Q1~Q5+LS的年化收益和夏普比率，4种方法叠加
3. **图3：分年度对比柱状图** — 2023 vs 2024的年化收益和夏普比率
4. **图4：Bootstrap CI误差棒图** — 各方法年化收益和夏普比率的点估计+95%CI
5. **图5：Top N敏感性折线图** — Top 5~50的年化收益和夏普比率变化
6. **图6：持有期敏感性折线图** — 1/2/4周调仓的年化收益和夏普比率变化
7. **图7：配对检验热力图** — 6x6 p值矩阵
8. **图8：DSR对比柱状图** — 各方法DSR值与0.95阈值线

#### Scenario: 图表生成
- **WHEN** 运行 `python scripts/generate_figures.py`
- **THEN** 在 `data/figures/` 目录下生成8张PNG图表，DPI=300，中文显示正常

### Requirement: 统一图表风格
所有图表 SHALL 使用统一的视觉风格：
- 配色：使用seaborn的Set2调色板
- 字体：中文使用SimHei或Microsoft YaHei
- 尺寸：单栏图8x6英寸，双栏图12x5英寸
- DPI：300
- 坐标轴标签、图例使用中文
