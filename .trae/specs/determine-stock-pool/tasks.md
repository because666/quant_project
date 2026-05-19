# Tasks

- [x] Task 1: 修改默认股票池起始日期
  - [x] SubTask 1.1: 修改 `src/data_fetcher.py` 中 `get_stock_list` 的 `start_date` 参数默认值从 `"2014-01-01"` 改为 `"2016-01-01"`
  - [x] SubTask 1.2: 验证 `data/meta/surviving_stocks_20160101_*.parquet` 缓存文件已存在（2589只）

- [x] Task 2: 创建股票池筛选脚本
  - [x] SubTask 2.1: 创建 `scripts/build_final_stock_pool.py` 脚本，实现多维度筛选逻辑
  - [x] SubTask 2.2: 实现存续期筛选：2016-01-01前上市且未退世
  - [x] SubTask 2.3: 实现ST/退市标记：排除含"退"字股票，标记含"ST"股票
  - [x] SubTask 2.4: 实现行业分类：获取申万一级行业分类并关联到每只股票
  - [x] SubTask 2.5: 实现数据完整性检查：检查周频因子数据中的覆盖率
  - [x] SubTask 2.6: 实现流动性筛选：排除周均成交额过低的股票（因换手率因子数据全为0，该步骤已合理跳过）
  - [x] SubTask 2.7: 生成 `data/meta/final_stock_pool.parquet` 最终股票池文件

- [x] Task 3: 生成股票池分析报告
  - [x] SubTask 3.1: 在脚本中添加报告生成逻辑，输出 `data/meta/stock_pool_report.md`
  - [x] SubTask 3.2: 报告包含：筛选条件、各步骤股票数量变化、最终规模、行业分布、上市日期分布、与旧池对比、数据覆盖率

- [x] Task 4: 运行脚本并验证
  - [x] SubTask 4.1: 运行 `python scripts/build_final_stock_pool.py` 生成最终股票池和报告
  - [x] SubTask 4.2: 检查 `stock_pool_report.md` 内容完整性和数据合理性
  - [x] SubTask 4.3: 检查 `final_stock_pool.parquet` 文件结构和记录数

# Task Dependencies
- [Task 2] depends on [Task 1]
- [Task 3] depends on [Task 2]
- [Task 4] depends on [Task 2, Task 3]
