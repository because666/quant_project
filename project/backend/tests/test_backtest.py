"""BacktestEngine 涨跌停过滤与交易成本公式单测。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.backtest import BacktestEngine, RandomPredictor

_BACKEND = Path(__file__).resolve().parents[1]


class _FakePredictor:
    model_type = "lightgbm"
    _factor_cols = ["f1"]
    _data_dir = _BACKEND / "data"

    def predict_panel(self, factor_df: pd.DataFrame) -> pd.DataFrame:
        if factor_df.empty:
            return pd.DataFrame(columns=["date", "stock_code", "score"])
        rank = {"AAA": 3.0, "BBB": 2.0, "CCC": 1.0}
        out = factor_df[["date", "stock_code"]].copy()
        out["date"] = pd.to_datetime(out["date"])
        out["stock_code"] = out["stock_code"].astype(str)
        out["score"] = out["stock_code"].map(rank).astype(float)
        return out

    def predict(self, factor_df: pd.DataFrame) -> pd.DataFrame:
        rank = {"AAA": 3.0, "BBB": 2.0, "CCC": 1.0}
        out = pd.DataFrame(
            {
                "stock_code": factor_df["stock_code"].astype(str),
                "score": factor_df["stock_code"].map(rank).astype(float),
                "f1": factor_df["f1"],
            }
        )
        return out.sort_values("score", ascending=False, kind="mergesort").reset_index(drop=True)


def test_limit_up_stock_not_in_target_holdings() -> None:
    """涨停标的不可买入，Top N 顺延至下一名。"""
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-05"] * 3),
            "stock_code": ["AAA", "BBB", "CCC"],
            "f1": [0.0, 0.0, 0.0],
            "close": [10.0, 10.0, 10.0],
            "buy_blocked_limit_up": [True, False, False],
            "sell_blocked_limit_down": [False, False, False],
            "future_return_1w": [0.01, 0.01, 0.01],
        }
    )
    eng = BacktestEngine("lightgbm", top_n=2, initial_capital=100_000.0, enable_limit_price=True)
    out = eng.run_backtest(df, predictor=_FakePredictor(), use_split="all")
    assert len(out) == 1
    h = json.loads(str(out.iloc[0]["holdings"]))
    assert "AAA" not in h
    assert "BBB" in h and "CCC" in h


def test_sell_cash_includes_commission_slippage_stamp() -> None:
    """卖出侧：滑点压低成交价，再扣双边佣金与卖出印花税。"""
    eng = BacktestEngine(
        "lightgbm",
        top_n=1,
        initial_capital=1.0,
        commission=0.0003,
        slippage=0.001,
        stamp_tax=0.0005,
    )
    # 100 股，收盘价 10：单价 9.99，成交额 999，再 * (1 - 0.0003 - 0.0005)
    c = eng._cash_from_sell(100.0, 10.0)
    assert abs(c - 999.0 * (1.0 - 0.0003 - 0.0005)) < 1e-6


def test_buy_reduces_nav_below_initial_due_to_fees() -> None:
    """首周满仓买入后总市值略低于初始现金（佣金+滑点）。"""
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-05"]),
            "stock_code": ["ZZZ"],
            "f1": [0.0],
            "close": [100.0],
            "buy_blocked_limit_up": [False],
            "sell_blocked_limit_down": [False],
            "future_return_1w": [0.0],
        }
    )

    class _P:
        model_type = "lightgbm"
        _factor_cols = ["f1"]
        _data_dir = _BACKEND / "data"

        def predict_panel(self, factor_df: pd.DataFrame) -> pd.DataFrame:
            out = factor_df[["date", "stock_code"]].copy()
            out["date"] = pd.to_datetime(out["date"])
            out["score"] = 1.0
            return out

        def predict(self, factor_df: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame({"stock_code": ["ZZZ"], "score": [1.0], "f1": [0.0]})

    eng = BacktestEngine("lightgbm", top_n=1, initial_capital=1_000_000.0)
    out = eng.run_backtest(df, predictor=_P(), use_split="all")
    tv = float(out.iloc[0]["total_value"])
    assert tv < 1_000_000.0
    # 约 1e6 / (1.001*1.0003) 股 * 100 ≈ 998701
    assert tv > 998_500.0


def test_predict_failure_skips_week_without_crash() -> None:
    """某周 predict 失败时跳过调仓，回测继续。"""

    class _BadPredictor:
        model_type = "lightgbm"
        _factor_cols = ["f1"]
        _data_dir = _BACKEND / "data"

        def predict_panel(self, factor_df: pd.DataFrame) -> pd.DataFrame:
            raise RuntimeError("batch down")

        def predict(self, factor_df: pd.DataFrame) -> pd.DataFrame:
            raise ValueError("no factors")

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-05", "2024-01-12"]),
            "stock_code": ["ZZZ", "ZZZ"],
            "f1": [0.0, 0.0],
            "close": [10.0, 10.0],
            "buy_blocked_limit_up": [False, False],
            "sell_blocked_limit_down": [False, False],
            "future_return_1w": [0.0, 0.0],
        }
    )
    eng = BacktestEngine("lightgbm", top_n=1, initial_capital=100_000.0)
    out = eng.run_backtest(df, predictor=_BadPredictor(), use_split="all")
    assert len(out) == 2
    assert all(float(out.iloc[i]["total_value"]) == pytest.approx(100_000.0) for i in range(2))


class TestRandomPredictor:
    """RandomPredictor 单元测试：正常流程、空输入、与 BacktestEngine 集成。"""

    def test_predict_returns_sorted_scores(self) -> None:
        """predict 方法返回按 score 降序排列的 DataFrame。"""
        rng = np.random.default_rng(123)
        pred = RandomPredictor(rng)
        panel_df = pd.DataFrame({"stock_code": ["A", "B", "C", "D", "E"]})
        result = pred.predict(panel_df)
        assert list(result.columns) == ["stock_code", "score"]
        assert len(result) == 5
        scores = result["score"].to_numpy()
        assert (scores[:-1] >= scores[1:]).all(), "score 应降序排列"
        assert (scores >= 0.0).all() and (scores < 1.0).all(), "score 应在 [0, 1) 范围内"

    def test_predict_empty_input(self) -> None:
        """predict 方法对空 DataFrame 返回空结果。"""
        rng = np.random.default_rng(0)
        pred = RandomPredictor(rng)
        result = pred.predict(pd.DataFrame(columns=["stock_code"]))
        assert result.empty
        assert list(result.columns) == ["stock_code", "score"]

    def test_predict_panel_multiple_dates(self) -> None:
        """predict_panel 方法按日期分组生成随机分数。"""
        rng = np.random.default_rng(42)
        pred = RandomPredictor(rng)
        factor_df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-05"] * 3 + ["2024-01-12"] * 3),
            "stock_code": ["A", "B", "C", "A", "B", "C"],
        })
        result = pred.predict_panel(factor_df)
        assert list(result.columns) == ["date", "stock_code", "score"]
        assert len(result) == 6
        dates_in_result = result["date"].unique()
        assert len(dates_in_result) == 2
        for dt in dates_in_result:
            sub = result[result["date"] == dt]
            scores = sub["score"].to_numpy()
            assert len(scores) == 3
            assert (scores >= 0.0).all() and (scores < 1.0).all()

    def test_predict_panel_empty_input(self) -> None:
        """predict_panel 方法对空 DataFrame 返回空结果。"""
        rng = np.random.default_rng(0)
        pred = RandomPredictor(rng)
        result = pred.predict_panel(pd.DataFrame(columns=["date", "stock_code"]))
        assert result.empty
        assert list(result.columns) == ["date", "stock_code", "score"]

    def test_predict_reproducible_with_same_seed(self) -> None:
        """相同种子的 RandomPredictor 产生相同的分数序列。"""
        pred1 = RandomPredictor(np.random.default_rng(999))
        pred2 = RandomPredictor(np.random.default_rng(999))
        panel_df = pd.DataFrame({"stock_code": ["X", "Y", "Z"]})
        r1 = pred1.predict(panel_df)
        r2 = pred2.predict(panel_df)
        pd.testing.assert_frame_equal(r1, r2)

    def test_predict_different_seeds_produce_different_scores(self) -> None:
        """不同种子的 RandomPredictor 产生不同的分数序列。"""
        pred1 = RandomPredictor(np.random.default_rng(1))
        pred2 = RandomPredictor(np.random.default_rng(2))
        panel_df = pd.DataFrame({"stock_code": ["X", "Y", "Z"]})
        r1 = pred1.predict(panel_df)
        r2 = pred2.predict(panel_df)
        assert not np.allclose(r1["score"].to_numpy(), r2["score"].to_numpy())

    def test_factor_cols_empty_when_data_dir_missing(self) -> None:
        """data_dir 不存在时 _factor_cols 为空列表，不抛异常。"""
        rng = np.random.default_rng(0)
        pred = RandomPredictor(rng, data_dir=Path("/nonexistent/path"))
        assert pred._factor_cols == []

    def test_model_type_is_lightgbm(self) -> None:
        """model_type 属性固定为 lightgbm。"""
        pred = RandomPredictor(np.random.default_rng(0))
        assert pred.model_type == "lightgbm"


def test_random_predictor_with_backtest_engine() -> None:
    """RandomPredictor 作为 custom_predictor 传入 BacktestEngine，回测正常完成并扣除交易成本。"""
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-05"] * 3 + ["2024-01-12"] * 3),
        "stock_code": ["AAA", "BBB", "CCC"] * 2,
        "f1": [0.0] * 6,
        "close": [10.0, 20.0, 30.0, 11.0, 21.0, 31.0],
        "buy_blocked_limit_up": [False] * 6,
        "sell_blocked_limit_down": [False] * 6,
        "future_return_1w": [0.01] * 6,
    })
    rng = np.random.default_rng(42)
    pred = RandomPredictor(rng)
    eng = BacktestEngine("lightgbm", top_n=2, initial_capital=1_000_000.0, custom_predictor=pred)
    out = eng.run_backtest(df, use_split="all")
    assert len(out) == 2
    assert float(out.iloc[0]["total_value"]) < 1_000_000.0, "首周买入后因交易成本，总市值应低于初始资金"


def test_random_predictor_nav_starts_at_one() -> None:
    """使用 RandomPredictor 回测后，NAV 序列首值为 1.0。"""
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-05"] * 2),
        "stock_code": ["AAA", "BBB"],
        "f1": [0.0, 0.0],
        "close": [10.0, 20.0],
        "buy_blocked_limit_up": [False, False],
        "sell_blocked_limit_down": [False, False],
        "future_return_1w": [0.0, 0.0],
    })
    rng = np.random.default_rng(7)
    pred = RandomPredictor(rng)
    eng = BacktestEngine("lightgbm", top_n=2, initial_capital=500_000.0, custom_predictor=pred)
    out = eng.run_backtest(df, use_split="all")
    nav = np.concatenate([[1.0], out["total_value"].to_numpy() / 500_000.0])
    assert nav[0] == pytest.approx(1.0)
    assert nav[-1] < 1.0, "首周买入后因交易成本，NAV 应低于 1.0"
