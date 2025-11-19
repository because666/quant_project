"""
量化策略主程序
组长负责维护这个文件，用来串联整个流程
"""

print("=== 量化选股策略项目开始运行 ===")

# 将来，这里会依次调用三个组的代码
# 1. 首先会调用数据组的代码
print("[将来这里] 步骤1: 执行数据获取与清洗...")
# import data.data_fetcher as data_fetcher
# data_fetcher.run()

# 2. 然后调用策略组的代码
print("[将来这里] 步骤2: 执行模型训练与信号生成...")
# import strategy.model_trainer as strategy
# strategy.run()

# 3. 最后调用回测组的代码
print("[将来这里] 步骤3: 执行回测验证...")
# import backtest.backtest_engine as backtest
# backtest.run()

print("=== 所有流程执行完毕 ===")