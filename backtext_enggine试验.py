import numpy as np
import pandas as pd

class BacktestEngine:
    def __init__(self, initial_capital=100000):
        """
        初始化回测引擎
        :param initial_capital: 初始资金，默认100000元
        """
        self.initial_capital = initial_capital  # 初始资金
        self.current_capital = initial_capital  # 当前资金
        self.daily_capital = []  # 记录每日总资产
        self.trades = []  # 记录每笔交易：(日期, 信号, 收益)

    def run(self, signals, prices):
        """
        核心运行函数：循环每一天，根据信号模拟买卖
        :param signals: 交易信号列表，1=买入，-1=卖出，0=空仓，长度与prices一致
        :param prices: 每日价格列表，长度与signals一致
        """
        if len(signals) != len(prices):
            raise ValueError("信号列表和价格列表长度必须一致")
        
        holding = False  # 是否持有仓位
        buy_price = 0  # 买入价格

        # 循环每一天处理信号
        for day in range(len(signals)):
            signal = signals[day]
            price = prices[day]

            # 处理买入信号
            if signal == 1 and not holding:
                buy_price = price
                holding = True
                print(f"第{day+1}天：买入，价格={price:.2f}")
            
            # 处理卖出信号
            elif signal == -1 and holding:
                profit = (price - buy_price) / buy_price * self.current_capital
                self.current_capital += profit
                self.trades.append((day+1, "sell", profit))
                holding = False
                print(f"第{day+1}天：卖出，价格={price:.2f}，单笔收益={profit:.2f}")
            
            # 记录当日总资产
            self.daily_capital.append(self.current_capital)

    def calculate_performance(self):
        """
        计算绩效指标：总收益率、最大回撤、胜率
        """
        if not self.daily_capital:
            raise RuntimeError("请先调用run()执行回测")
        
        # 1. 总收益率
        total_return = (self.current_capital - self.initial_capital) / self.initial_capital
        
        # 2. 最大回撤
        capital_array = np.array(self.daily_capital)
        peak = np.maximum.accumulate(capital_array)
        drawdown = (capital_array - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # 3. 胜率（盈利交易占比）
        if not self.trades:
            win_rate = 0
        else:
            win_trades = [trade for trade in self.trades if trade[2] > 0]
            win_rate = len(win_trades) / len(self.trades)
        
        # 清晰打印结果
        print("\n==================== 回测绩效指标 ====================")
        print(f"初始资金：{self.initial_capital:.2f} 元")
        print(f"最终资金：{self.current_capital:.2f} 元")
        print(f"总收益率：{total_return:.2%}")
        print(f"最大回撤：{max_drawdown:.2%}")
        print(f"交易胜率：{win_rate:.2%}（盈利交易{len(win_trades)}/{len(self.trades)}笔）")
        
        return {
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate
        }

# 测试代码
if __name__ == "__main__":
    # 模拟信号（1=买入，-1=卖出，0=空仓）
    test_signals = [1, 0, 0, -1, 1, 0, -1, 0, 1, -1]
    # 模拟价格数据
    test_prices = [10, 10.2, 10.5, 10.8, 10.6, 10.9, 11.2, 11.0, 11.5, 12.0]
    
    # 初始化并运行回测
    engine = BacktestEngine(initial_capital=100000)
    engine.run(test_signals, test_prices)
    # 计算并输出绩效
    engine.calculate_performance()
import numpy as np
import pandas as pd

class BacktestEngine:
    def __init__(self, initial_capital=100000):
        """
        初始化回测引擎
        :param initial_capital: 初始资金，默认100000元
        """
        self.initial_capital = initial_capital  # 初始资金
        self.current_capital = initial_capital  # 当前资金
        self.daily_capital = []  # 记录每日总资产
        self.trades = []  # 记录每笔交易：(日期, 信号, 收益)

    def run(self, signals, prices):
        """
        核心运行函数：循环每一天，根据信号模拟买卖
        :param signals: 交易信号列表，1=买入，-1=卖出，0=空仓，长度与prices一致
        :param prices: 每日价格列表，长度与signals一致
        """
        if len(signals) != len(prices):
            raise ValueError("信号列表和价格列表长度必须一致")
        
        holding = False  # 是否持有仓位
        buy_price = 0  # 买入价格

        # 循环每一天处理信号
        for day in range(len(signals)):
            signal = signals[day]
            price = prices[day]

            # 处理买入信号
            if signal == 1 and not holding:
                buy_price = price
                holding = True
                print(f"第{day+1}天：买入，价格={price:.2f}")
            
            # 处理卖出信号
            elif signal == -1 and holding:
                profit = (price - buy_price) / buy_price * self.current_capital
                self.current_capital += profit
                self.trades.append((day+1, "sell", profit))
                holding = False
                print(f"第{day+1}天：卖出，价格={price:.2f}，单笔收益={profit:.2f}")
            
            # 记录当日总资产
            self.daily_capital.append(self.current_capital)

    def calculate_performance(self):
        """
        计算绩效指标：总收益率、最大回撤、胜率
        """
        if not self.daily_capital:
            raise RuntimeError("请先调用run()执行回测")
        
        # 1. 总收益率
        total_return = (self.current_capital - self.initial_capital) / self.initial_capital
        
        # 2. 最大回撤
        capital_array = np.array(self.daily_capital)
        peak = np.maximum.accumulate(capital_array)
        drawdown = (capital_array - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # 3. 胜率（盈利交易占比）
        if not self.trades:
            win_rate = 0
        else:
            win_trades = [trade for trade in self.trades if trade[2] > 0]
            win_rate = len(win_trades) / len(self.trades)
        
        # 清晰打印结果
        print("\n==================== 回测绩效指标 ====================")
        print(f"初始资金：{self.initial_capital:.2f} 元")
        print(f"最终资金：{self.current_capital:.2f} 元")
        print(f"总收益率：{total_return:.2%}")
        print(f"最大回撤：{max_drawdown:.2%}")
        print(f"交易胜率：{win_rate:.2%}（盈利交易{len(win_trades)}/{len(self.trades)}笔）")
        
        return {
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate
        }

# 测试代码
if __name__ == "__main__":
    # 模拟信号（1=买入，-1=卖出，0=空仓）
    test_signals = [1, 0, 0, -1, 1, 0, -1, 0, 1, -1]
    # 模拟价格数据
    test_prices = [10, 10.2, 10.5, 10.8, 10.6, 10.9, 11.2, 11.0, 11.5, 12.0]
    
    # 初始化并运行回测
    engine = BacktestEngine(initial_capital=100000)
    engine.run(test_signals, test_prices)
    # 计算并输出绩效
    engine.calculate_performance()
