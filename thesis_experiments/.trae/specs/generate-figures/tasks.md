# Tasks

- [x] Task 1: 创建图表生成脚本框架
  - [x] 1.1: 创建 `scripts/generate_figures.py`，包含统一的绘图风格配置（字体、配色、DPI）
  - [x] 1.2: 实现JSON数据加载函数，读取所有实验结果文件
  - [x] 1.3: 确认 matplotlib 和 seaborn 已安装

- [x] Task 2: 生成图1-综合对比分组柱状图
  - [x] 2.1: 读取 experiment4_summary.json，绘制4种方法的年化收益/夏普/回撤分组柱状图

- [x] Task 3: 生成图2-分位数单调性折线图
  - [x] 3.1: 读取 quantile_analysis.json，绘制Q1~Q5+LS的年化收益折线图（4种方法叠加）

- [x] Task 4: 生成图3-分年度对比柱状图
  - [x] 4.1: 读取 yearly_analysis.json，绘制2023 vs 2024的年化收益分组柱状图

- [x] Task 5: 生成图4-Bootstrap CI误差棒图
  - [x] 5.1: 读取 bootstrap_ci.json，绘制各方法年化收益和夏普的点估计+95%CI误差棒图

- [x] Task 6: 生成图5-Top N敏感性折线图
  - [x] 6.1: 读取 sensitivity_topn.json，绘制Top 5~50的年化收益和夏普变化折线图

- [x] Task 7: 生成图6-持有期敏感性折线图
  - [x] 7.1: 读取 sensitivity_holding.json，绘制1/2/4周调仓的年化收益和夏普变化折线图

- [x] Task 8: 生成图7-配对检验热力图
  - [x] 8.1: 读取 paired_tests.json，绘制6x6 p值热力图

- [x] Task 9: 生成图8-DSR对比柱状图
  - [x] 9.1: 读取 deflated_sharpe.json，绘制各方法DSR值与0.95阈值线

- [x] Task 10: 运行脚本并验证图表
  - [x] 10.1: 运行 `python scripts/generate_figures.py`，确认8张图表生成
  - [x] 10.2: 检查图表中文显示、数据准确性、视觉风格统一性

# Task Dependencies
- Task 10 depends on Task 1-9
- Task 2-9 可并行（各自读取不同JSON文件）
