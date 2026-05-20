from src.backtest import BacktestEngine

eng = BacktestEngine("lightgbm", top_n=10, rebalance_freq=2)
assert eng._rebalance_freq == 2, f"rebalance_freq=2 failed, got {eng._rebalance_freq}"

eng2 = BacktestEngine("lightgbm", top_n=10)
assert eng2._rebalance_freq == 1, f"default rebalance_freq failed, got {eng2._rebalance_freq}"

try:
    BacktestEngine("lightgbm", top_n=10, rebalance_freq=0)
    assert False, "rebalance_freq=0 should raise ValueError"
except ValueError:
    pass

try:
    BacktestEngine("lightgbm", top_n=10, rebalance_freq=-1)
    assert False, "rebalance_freq=-1 should raise ValueError"
except ValueError:
    pass

from src.backtest import _engine_params_snapshot
eng3 = BacktestEngine("lightgbm", top_n=10, rebalance_freq=4)
params = _engine_params_snapshot(eng3)
assert params["rebalance_freq"] == 4, f"params rebalance_freq failed, got {params['rebalance_freq']}"

print("ALL TESTS PASSED")
