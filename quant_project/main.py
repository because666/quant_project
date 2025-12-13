# -*- coding: utf-8 -*-
"""
量化选股策略主程序

该主程序整合了量化选股策略的所有模块，包括：
1. 数据获取与清洗
2. 模型训练
3. 交易信号生成
4. 回测验证

通过运行该主程序，可以完整执行量化选股策略的整个流程，并生成回测结果报告。
"""

import config as cfg
from data.data_fetcher import fetch_data
from data.data_cleaner import clean_data
from strategy.model_trainer import train_model
from strategy.signal_generator import generate_signals
from backtest.backtest_engine import run_backtest

def main():
    """
    主程序入口函数，执行量化选股策略的完整流程
    
    参数:
        None
    
    返回:
        None
    """
    print("📊 === 量化选股策略项目开始运行 ===")
    
    try:
        # 步骤1: 数据获取与清洗
        print("\n🔍 步骤1: 执行数据获取与清洗...")
        fetch_data(cfg.STOCK_LIST, cfg.TRAIN_START_DATE, cfg.TRAIN_END_DATE)
        clean_data()
        print("✅ 数据获取与清洗完成")

        # 步骤2: 模型训练
        print("\n🤖 步骤2: 执行模型训练...")
        model = train_model()
        print("✅ 模型训练完成")

        # 步骤3: 交易信号生成
        print("\n📈 步骤3: 生成交易信号...")
        generate_signals(model)
        print("✅ 交易信号生成完成")

        # 步骤4: 回测验证
        print("\n📊 步骤4: 执行回测验证...")
        performance = run_backtest()
        print("✅ 回测验证完成")

        print("\n🎉 === 所有流程执行完毕 ===")
        return performance
        
    except Exception as e:
        print(f"\n❌ 程序执行失败: {e}")
        raise


def run_specific_step(step):
    """
    运行特定步骤的函数，用于调试和测试
    
    参数:
        step (int): 要运行的步骤
                   1: 数据获取与清洗
                   2: 模型训练
                   3: 交易信号生成
                   4: 回测验证
    
    返回:
        None
    """
    print(f"📋 运行特定步骤: {step}")
    
    try:
        if step == 1:
            # 步骤1: 数据获取与清洗
            print("\n🔍 执行数据获取与清洗...")
            fetch_data(cfg.STOCK_LIST, cfg.TRAIN_START_DATE, cfg.TRAIN_END_DATE)
            clean_data()
            print("✅ 数据获取与清洗完成")
        
        elif step == 2:
            # 步骤2: 模型训练
            print("\n🤖 执行模型训练...")
            model = train_model()
            print("✅ 模型训练完成")
        
        elif step == 3:
            # 步骤3: 交易信号生成
            print("\n📈 生成交易信号...")
            generate_signals()
            print("✅ 交易信号生成完成")
        
        elif step == 4:
            # 步骤4: 回测验证
            print("\n📊 执行回测验证...")
            performance = run_backtest()
            print("✅ 回测验证完成")
        
        else:
            print(f"❌ 无效的步骤: {step}")
            print("请输入有效的步骤编号: 1-4")
            
    except Exception as e:
        print(f"\n❌ 执行步骤{step}失败: {e}")
        raise


if __name__ == "__main__":
    """
    主程序入口
    
    运行方式:
    1. 直接运行: 执行完整的量化选股策略流程
    2. 调试模式: 可以通过修改下方的step参数，运行特定步骤
    """
    # 选择运行模式
    run_full_process = True  # True: 运行完整流程, False: 运行特定步骤
    step_to_run = 1  # 当run_full_process为False时，运行的特定步骤
    
    if run_full_process:
        main()
    else:
        run_specific_step(step_to_run)

