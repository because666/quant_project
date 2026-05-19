"""metrics 模块：净值指标、换手与胜率、NDCG、汇总字典可 JSON。"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    aggregate_metrics,
    annualized_return,
    bootstrap_metric,
    calmar_ratio,
    factor_ic,
    factor_icir,
    max_drawdown,
    mean_average_precision,
    metrics_by_year,
    ndcg_at_k,
    paired_significance_test,
    quantile_portfolio_returns,
    sharpe_ratio,
    sortino_ratio,
    turnover_rate,
    weekly_nav_to_daily_business_ffill,
    win_rate,
)


def test_annualized_return_positive_trend() -> None:
    idx = pd.date_range("2024-01-02", periods=252, freq="B")
    nav = pd.Series(np.linspace(100.0, 110.0, len(idx)), index=idx)
    ar = annualized_return(nav)
    assert ar > 0
    assert ar < 2.0


def test_weekly_nav_to_daily_ffill_length() -> None:
    idx = pd.to_datetime(["2024-01-05", "2024-01-12"])
    nav = pd.Series([100.0, 101.0], index=idx)
    d = weekly_nav_to_daily_business_ffill(nav)
    assert len(d) >= 5
    assert d.iloc[0] == 100.0
    assert d.iloc[-1] == 101.0


def test_nav_forward_fill() -> None:
    idx = pd.date_range("2024-01-02", periods=10, freq="B")
    nav = pd.Series([100.0, np.nan, np.nan, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0], index=idx)
    ar = annualized_return(nav)
    assert not np.isnan(ar)


def test_max_drawdown_peak_trough_order() -> None:
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    nav = pd.Series([100.0, 110.0, 80.0, 100.0], index=idx)
    dd = max_drawdown(nav)
    assert dd["max_drawdown"] == pytest.approx((110.0 - 80.0) / 110.0, rel=1e-9)
    assert dd["peak_date"] == "2024-01-03"
    assert dd["trough_date"] == "2024-01-04"


def test_sharpe_uses_log_returns_finite() -> None:
    idx = pd.date_range("2024-01-02", periods=252, freq="B")
    rng = np.random.default_rng(42)
    nav = pd.Series(100 * np.cumprod(1 + rng.normal(0.0003, 0.008, size=len(idx))), index=idx)
    s = sharpe_ratio(nav, risk_free_rate=0.03)
    assert np.isfinite(s)


def test_turnover_aggregate_matches_side_aggregate() -> None:
    agg = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-05", "2024-01-12"]),
            "buy_amount": [400_000.0, 0.0],
            "sell_amount": [100_000.0, 0.0],
            "total_value": [1_000_000.0, 1_000_000.0],
        }
    )
    side = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-05", "2024-01-05", "2024-01-12"]),
            "side": ["buy", "sell", "buy"],
            "gross_amount": [400_000.0, 100_000.0, 0.0],
            "total_value": [1_000_000.0, 1_000_000.0, 1_000_000.0],
        }
    )
    t1 = turnover_rate(agg)
    t2 = turnover_rate(side)
    assert t1 == pytest.approx(t2, rel=1e-9)
    assert t1 == pytest.approx(0.25, rel=1e-9)


def test_turnover_matches_manual_rebalance_notion() -> None:
    """单周：买+卖名义 / 当日总资产 = (3e5+3e5)/1e6 = 0.6。"""
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-02-01"]),
            "buy_amount": [300_000.0],
            "sell_amount": [300_000.0],
            "total_value": [1_000_000.0],
        }
    )
    assert turnover_rate(df) == pytest.approx(0.6, rel=1e-9)


def test_win_rate() -> None:
    df = pd.DataFrame({"pnl": [100.0, -50.0, 0.0, 20.0]})
    assert win_rate(df) == pytest.approx(0.5)


def test_win_rate_from_nav_series_matches_weekly_portfolio() -> None:
    """与回测 trades_df（调仓日 total_value）一致：周度收益为正的占比。"""
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-05", "2024-01-12", "2024-01-19"]),
            "buy_amount": [0.0, 100_000.0, 0.0],
            "sell_amount": [0.0, 0.0, 50_000.0],
            "total_value": [1_000_000.0, 1_050_000.0, 1_030_000.0],
        }
    )
    # 收益: +5%, -1.9% -> 一正一负
    assert win_rate(df) == pytest.approx(1.0 / 2.0)


def test_ndcg_perfect_one() -> None:
    y_true = np.array([3.0, 2.0, 1.0, 0.0])
    y_score = np.array([4.0, 3.0, 2.0, 1.0])
    assert ndcg_at_k(y_true, y_score, k=4) == pytest.approx(1.0, rel=1e-9)


def test_aggregate_metrics_json_roundtrip() -> None:
    idx = pd.date_range("2024-01-02", periods=60, freq="B")
    nav = pd.Series(100 * (1.001 ** np.arange(len(idx))), index=idx)
    m = aggregate_metrics(
        nav,
        trades_df=pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-10"]),
                "buy_amount": [50_000.0],
                "sell_amount": [10_000.0],
                "total_value": [100_000.0],
            }
        ),
        y_true_ndcg=[2, 1, 0],
        y_score_ndcg=[0.3, 0.2, 0.1],
        ndcg_k=3,
    )
    s = json.dumps(m)
    back = json.loads(s)
    assert "annualized_return" in back
    assert back["drawdown_peak_date"] is None or isinstance(back["drawdown_peak_date"], str)


def test_calmar_ratio_normal() -> None:
    """Calmar比率：年化收益/最大回撤。"""
    idx = pd.date_range("2024-01-02", periods=252, freq="B")
    rng = np.random.default_rng(42)
    nav = pd.Series(100 * np.cumprod(1 + rng.normal(0.0003, 0.008, size=len(idx))), index=idx)
    cr = calmar_ratio(nav)
    assert np.isfinite(cr)


def test_calmar_ratio_zero_drawdown() -> None:
    """最大回撤为0时返回NaN。"""
    idx = pd.date_range("2024-01-02", periods=10, freq="B")
    nav = pd.Series(np.linspace(100.0, 110.0, len(idx)), index=idx)
    cr = calmar_ratio(nav)
    assert np.isnan(cr) or np.isinf(cr)


def test_sortino_ratio_normal() -> None:
    """Sortino比率正常计算。"""
    idx = pd.date_range("2024-01-02", periods=252, freq="B")
    rng = np.random.default_rng(42)
    nav = pd.Series(100 * np.cumprod(1 + rng.normal(0.0003, 0.008, size=len(idx))), index=idx)
    sr = sortino_ratio(nav, risk_free_rate=0.03)
    assert np.isfinite(sr)


def test_sortino_ratio_no_downside() -> None:
    """无下行波动时返回NaN。"""
    idx = pd.date_range("2024-01-02", periods=10, freq="B")
    nav = pd.Series(100 * (1.001 ** np.arange(len(idx))), index=idx)
    sr = sortino_ratio(nav, risk_free_rate=0.0)
    assert np.isnan(sr)


def test_factor_ic_global() -> None:
    """全局IC（Spearman秩相关系数）。"""
    rng = np.random.default_rng(42)
    factor = rng.normal(0, 1, size=100)
    ret = 0.5 * factor + rng.normal(0, 0.5, size=100)
    ic = factor_ic(factor, ret)
    assert isinstance(ic, float)
    assert ic > 0.3


def test_factor_ic_grouped() -> None:
    """按期分组IC。"""
    rng = np.random.default_rng(42)
    factor = rng.normal(0, 1, size=60)
    ret = 0.3 * factor + rng.normal(0, 0.5, size=60)
    groups = np.repeat(["A", "B", "C"], 20)
    ics = factor_ic(factor, ret, groupby=groups)
    assert isinstance(ics, np.ndarray)
    assert len(ics) == 3


def test_factor_ic_short_input() -> None:
    """输入不足3个时返回NaN。"""
    ic = factor_ic([1.0, 2.0], [3.0, 4.0])
    assert np.isnan(ic)


def test_factor_icir_normal() -> None:
    """ICIR正常计算。"""
    ics = np.array([0.05, 0.08, 0.03, 0.06, 0.07])
    icir = factor_icir(ics)
    assert np.isfinite(icir)
    assert icir > 0


def test_factor_icir_zero_std() -> None:
    """IC标准差为0时返回NaN。"""
    ics = np.array([0.05, 0.05, 0.05])
    icir = factor_icir(ics)
    assert np.isnan(icir)


def test_mean_average_precision_perfect() -> None:
    """完美排序MAP=1.0。"""
    y_true = [[3.0, 2.0, 1.0, 0.0]]
    y_score = [[4.0, 3.0, 2.0, 1.0]]
    map_val = mean_average_precision(y_true, y_score)
    assert map_val == pytest.approx(1.0, rel=1e-9)


def test_mean_average_precision_multiple_queries() -> None:
    """多query MAP。"""
    y_true = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    y_score = [[1.0, 0.5, 0.0], [0.8, 0.2, 0.1]]
    map_val = mean_average_precision(y_true, y_score)
    assert 0.0 <= map_val <= 1.0


def test_mean_average_precision_empty() -> None:
    """空输入返回NaN。"""
    map_val = mean_average_precision([], [])
    assert np.isnan(map_val)


def test_bootstrap_metric_contains_point() -> None:
    """Bootstrap置信区间包含点估计。"""
    rng = np.random.default_rng(42)
    returns = rng.normal(0.001, 0.02, size=100)
    result = bootstrap_metric(returns, np.mean, n_bootstrap=500, seed=42)
    assert "point" in result
    assert "lower" in result
    assert "upper" in result
    assert result["lower"] <= result["point"] <= result["upper"]


def test_bootstrap_metric_short_input() -> None:
    """输入不足2个时返回NaN。"""
    result = bootstrap_metric(np.array([0.01]), np.mean)
    assert np.isnan(result["point"])


def test_paired_significance_test_same_distribution() -> None:
    """相同分布不显著。"""
    rng = np.random.default_rng(42)
    a = rng.normal(0.01, 0.02, size=100)
    b = a + rng.normal(0, 0.001, size=100)
    result = paired_significance_test(a, b, alpha=0.05)
    assert "t_stat" in result
    assert "p_value" in result
    assert "significant" in result
    assert "method" in result
    assert result["significant"] is False


def test_paired_significance_test_different_distribution() -> None:
    """不同分布显著。"""
    rng = np.random.default_rng(42)
    a = rng.normal(0.05, 0.02, size=100)
    b = rng.normal(0.0, 0.02, size=100)
    result = paired_significance_test(a, b, alpha=0.05)
    assert result["significant"] is True


def test_metrics_by_year_normal() -> None:
    """分年度指标计算。"""
    idx = pd.date_range("2022-01-03", periods=504, freq="B")
    rng = np.random.default_rng(42)
    nav = pd.Series(100 * np.cumprod(1 + rng.normal(0.0003, 0.008, size=len(idx))), index=idx)
    df = metrics_by_year(nav)
    assert not df.empty
    assert "annualized_return" in df.columns
    assert "sharpe_ratio" in df.columns
    assert "max_drawdown" in df.columns
    assert len(df) >= 2


def test_metrics_by_year_empty() -> None:
    """空输入返回空DataFrame。"""
    df = metrics_by_year(pd.Series([], dtype=float))
    assert df.empty


def test_quantile_portfolio_returns_normal() -> None:
    """分位数组合收益正常计算。"""
    rng = np.random.default_rng(42)
    dates = np.repeat(pd.date_range("2024-01-05", periods=4, freq="W-FRI"), 20)
    df = pd.DataFrame({
        "date": dates,
        "stock_code": [f"S{i:03d}" for i in range(80)],
        "score": rng.normal(0, 1, size=80),
        "future_return_1w": rng.normal(0.001, 0.02, size=80),
    })
    result = quantile_portfolio_returns(df, n_quantiles=5)
    assert "Q1" in result
    assert "Q5" in result
    assert "LS" in result
    assert len(result["Q1"]) == 4
    np.testing.assert_allclose(result["LS"], result["Q1"] - result["Q5"], atol=1e-10)


def test_quantile_portfolio_returns_empty() -> None:
    """空输入返回空字典。"""
    result = quantile_portfolio_returns(pd.DataFrame(), n_quantiles=5)
    assert result == {}


def test_quantile_portfolio_returns_vol_penalty_with_volatility() -> None:
    """vol_penalty > 0 且存在波动率列时，使用调整后得分分组。"""
    rng = np.random.default_rng(42)
    dates = np.repeat(pd.date_range("2024-01-05", periods=4, freq="W-FRI"), 20)
    df = pd.DataFrame({
        "date": dates,
        "stock_code": [f"S{i:03d}" for i in range(80)],
        "score": rng.normal(0, 1, size=80),
        "future_return_1w": rng.normal(0.001, 0.02, size=80),
        "volatility_12w": rng.uniform(0.1, 0.5, size=80),
    })
    result_no_penalty = quantile_portfolio_returns(df, n_quantiles=5, vol_penalty=0.0)
    result_with_penalty = quantile_portfolio_returns(df, n_quantiles=5, vol_penalty=1.0)
    assert "Q1" in result_with_penalty
    assert "Q5" in result_with_penalty
    assert "LS" in result_with_penalty
    assert len(result_with_penalty["Q1"]) == 4
    with_penalty = result_with_penalty["Q1"]
    without_penalty = result_no_penalty["Q1"]
    assert not np.allclose(with_penalty, without_penalty)


def test_quantile_portfolio_returns_vol_penalty_missing_col_warning(caplog: pytest.LogCaptureFixture) -> None:
    """vol_penalty > 0 但波动率列不存在时，发出警告并回退到原始 score。"""
    rng = np.random.default_rng(42)
    dates = np.repeat(pd.date_range("2024-01-05", periods=4, freq="W-FRI"), 20)
    df = pd.DataFrame({
        "date": dates,
        "stock_code": [f"S{i:03d}" for i in range(80)],
        "score": rng.normal(0, 1, size=80),
        "future_return_1w": rng.normal(0.001, 0.02, size=80),
    })
    result_no_penalty = quantile_portfolio_returns(df, n_quantiles=5, vol_penalty=0.0)
    with caplog.at_level("WARNING"):
        result_with_penalty = quantile_portfolio_returns(df, n_quantiles=5, vol_penalty=1.0)
    assert any("volatility_12w" in rec.message for rec in caplog.records)
    for q in ["Q1", "Q2", "Q3", "Q4", "Q5", "LS"]:
        np.testing.assert_allclose(result_with_penalty[q], result_no_penalty[q], atol=1e-10)


def test_quantile_portfolio_returns_vol_penalty_custom_col() -> None:
    """vol_penalty 使用自定义波动率列名。"""
    rng = np.random.default_rng(42)
    dates = np.repeat(pd.date_range("2024-01-05", periods=4, freq="W-FRI"), 20)
    df = pd.DataFrame({
        "date": dates,
        "stock_code": [f"S{i:03d}" for i in range(80)],
        "score": rng.normal(0, 1, size=80),
        "future_return_1w": rng.normal(0.001, 0.02, size=80),
        "my_vol": rng.uniform(0.1, 0.5, size=80),
    })
    result = quantile_portfolio_returns(df, n_quantiles=5, vol_penalty=0.5, volatility_col="my_vol")
    assert "Q1" in result
    assert "LS" in result
    assert len(result["Q1"]) == 4


def test_quantile_portfolio_returns_vol_penalty_zero_backward_compat() -> None:
    """vol_penalty=0.0 时行为与原始函数完全一致（向后兼容）。"""
    rng = np.random.default_rng(42)
    dates = np.repeat(pd.date_range("2024-01-05", periods=4, freq="W-FRI"), 20)
    df = pd.DataFrame({
        "date": dates,
        "stock_code": [f"S{i:03d}" for i in range(80)],
        "score": rng.normal(0, 1, size=80),
        "future_return_1w": rng.normal(0.001, 0.02, size=80),
        "volatility_12w": rng.uniform(0.1, 0.5, size=80),
    })
    result_default = quantile_portfolio_returns(df, n_quantiles=5)
    result_explicit_zero = quantile_portfolio_returns(df, n_quantiles=5, vol_penalty=0.0)
    for q in ["Q1", "Q2", "Q3", "Q4", "Q5", "LS"]:
        np.testing.assert_allclose(result_default[q], result_explicit_zero[q], atol=1e-10)


def test_aggregate_metrics_includes_calmar_sortino() -> None:
    """aggregate_metrics包含calmar_ratio和sortino_ratio。"""
    idx = pd.date_range("2024-01-02", periods=252, freq="B")
    rng = np.random.default_rng(42)
    nav = pd.Series(100 * np.cumprod(1 + rng.normal(0.0003, 0.008, size=len(idx))), index=idx)
    m = aggregate_metrics(nav)
    assert "calmar_ratio" in m
    assert "sortino_ratio" in m
