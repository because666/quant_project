"""反转排序回测验证"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest import BacktestEngine, compute_backtest_metrics

weekly_df = BacktestEngine("lightgbm", 20, 1_000_000.0).load_weekly_data()

# 正常排序（Top 20）
eng1 = BacktestEngine("lightgbm", 20, 1_000_000.0, reverse_sort=False)
result1 = eng1.run_backtest(weekly_df, use_split="test")
m1 = compute_backtest_metrics(result1, weekly_df, extended=False) if not result1.empty else {}

# 反转排序（Bottom 20）
eng2 = BacktestEngine("lightgbm", 20, 1_000_000.0, reverse_sort=True)
result2 = eng2.run_backtest(weekly_df, use_split="test")
m2 = compute_backtest_metrics(result2, weekly_df, extended=False) if not result2.empty else {}

print("=== 反转排序回测对比 ===")
ar1 = m1.get("annualized_return", 0) * 100
sr1 = m1.get("sharpe_ratio", 0)
dd1 = m1.get("max_drawdown", 0) * 100
ar2 = m2.get("annualized_return", 0) * 100
sr2 = m2.get("sharpe_ratio", 0)
dd2 = m2.get("max_drawdown", 0) * 100
print(f"Top 20 (正常): 年化={ar1:.2f}%, 夏普={sr1:.3f}, 回撤={dd1:.2f}%")
print(f"Bottom 20 (反转): 年化={ar2:.2f}%, 夏普={sr2:.3f}, 回撤={dd2:.2f}%")

# XGBoost同样测试
eng3 = BacktestEngine("xgboost", 20, 1_000_000.0, reverse_sort=False)
result3 = eng3.run_backtest(weekly_df, use_split="test")
m3 = compute_backtest_metrics(result3, weekly_df, extended=False) if not result3.empty else {}

eng4 = BacktestEngine("xgboost", 20, 1_000_000.0, reverse_sort=True)
result4 = eng4.run_backtest(weekly_df, use_split="test")
m4 = compute_backtest_metrics(result4, weekly_df, extended=False) if not result4.empty else {}

ar3 = m3.get("annualized_return", 0) * 100
sr3 = m3.get("sharpe_ratio", 0)
dd3 = m3.get("max_drawdown", 0) * 100
ar4 = m4.get("annualized_return", 0) * 100
sr4 = m4.get("sharpe_ratio", 0)
dd4 = m4.get("max_drawdown", 0) * 100
print(f"XGB Top 20 (正常): 年化={ar3:.2f}%, 夏普={sr3:.3f}, 回撤={dd3:.2f}%")
print(f"XGB Bottom 20 (反转): 年化={ar4:.2f}%, 夏普={sr4:.3f}, 回撤={dd4:.2f}%")
