# main.py - 你的主程序框架

import config as cfg
from data.data_fetcher import fetch_data
from data.data_cleaner import clean_data
from strategy.model_trainer import train_model
from strategy.signal_generator import generate_signals
from backtest.backtest_engine import run_backtest

def main():
    print("=== 量化选股策略项目开始运行 ===")
    
    # 步骤1: 数据获取与清洗
    print("步骤1: 执行数据获取与清洗...")
    try:
        
         fetch_data(cfg.STOCK_LIST, cfg.TRAIN_START_DATE, cfg.TRAIN_END_DATE)
         clean_data()
        print("✅ [待实现] 数据获取与清洗")
    except Exception as e:
        print(f"❌ 数据获取与清洗失败: {e}")
        return

    # 步骤2: 模型训练与信号生成
    print("步骤2: 执行模型训练与信号生成...")
    try:
        
        model = train_model()
        generate_signals(model)
        print("✅ [待实现] 模型训练与信号生成")
    except Exception as e:
        print(f"❌ 模型训练与信号生成失败: {e}")
        return

    # 步骤3: 回测验证
    print("步骤3: 执行回测验证...")
    try:
        
        performance = run_backtest()
        print("回测结果:", performance)
        print("✅ [待实现] 回测验证")
    except Exception as e:
        print(f"❌ 回测验证失败: {e}")
        return

    print("=== 所有流程执行完毕 ===")

if __name__ == "__main__":
    main()

