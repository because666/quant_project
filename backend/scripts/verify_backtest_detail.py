"""
精确定位回测收益与future_return_1w的差异来源。

核心问题：
- 持仓股票平均future_return_1w = +0.11%/周
- 回测组合平均周收益 = -0.41%/周
- 差异 = -0.52%/周，远超交易成本估算

排查方向：
1. 交易成本影响（零成本回测对比）
2. 现金拖累（未完全投资）
3. 持仓计算误差
4. 合成close价格是否正确
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.backtest import BacktestEngine, compute_backtest_metrics
from src.data_loader import split_by_time


def run_zero_cost_backtest() -> None:
    """零交易成本回测，排除成本影响。"""
    print("=" * 60)
    print("测试1：零交易成本回测")
    print("=" * 60)

    eng = BacktestEngine(
        "lightgbm", 20, 1_000_000.0,
        commission=0.0,
        slippage=0.0,
        stamp_tax=0.0,
    )
    weekly_df = eng.load_weekly_data()
    result_df = eng.run_backtest(weekly_df, use_split="test")

    if result_df.empty:
        print("回测结果为空")
        return

    m = compute_backtest_metrics(result_df, weekly_df, extended=False)
    print(f"零成本回测: 年化={m.get('annualized_return', 0) * 100:.2f}%, 夏普={m.get('sharpe_ratio', 0):.3f}")

    eng2 = BacktestEngine("lightgbm", 20, 1_000_000.0)
    result_df2 = eng2.run_backtest(weekly_df, use_split="test")
    m2 = compute_backtest_metrics(result_df2, weekly_df, extended=False)
    print(f"有成本回测: 年化={m2.get('annualized_return', 0) * 100:.2f}%, 夏普={m2.get('sharpe_ratio', 0):.3f}")
    print(f"成本拖累: {(m.get('annualized_return', 0) - m2.get('annualized_return', 0)) * 100:.2f}%")


def check_cash_drag() -> None:
    """检查现金拖累：回测中现金占总权益的比例。"""
    print()
    print("=" * 60)
    print("测试2：现金拖累分析")
    print("=" * 60)

    eng = BacktestEngine("lightgbm", 20, 1_000_000.0)
    weekly_df = eng.load_weekly_data()
    result_df = eng.run_backtest(weekly_df, use_split="test")

    if result_df.empty:
        print("回测结果为空")
        return

    result_df["cash_ratio"] = result_df["cash"] / result_df["total_value"]
    avg_cash_ratio = float(result_df["cash_ratio"].mean())
    max_cash_ratio = float(result_df["cash_ratio"].max())
    min_cash_ratio = float(result_df["cash_ratio"].min())

    print(f"平均现金占比: {avg_cash_ratio * 100:.2f}%")
    print(f"最大现金占比: {max_cash_ratio * 100:.2f}%")
    print(f"最小现金占比: {min_cash_ratio * 100:.2f}%")

    if avg_cash_ratio > 0.1:
        print("⚠️ 现金占比过高！大量资金未投入股市，造成严重现金拖累")
    else:
        print("✅ 现金占比正常")


def check_synthetic_close_correctness() -> None:
    """精确验证合成close价格：close[t+1]/close[t]-1 应等于 future_return_1w[t]。"""
    print()
    print("=" * 60)
    print("测试3：合成close价格正确性验证")
    print("=" * 60)

    eng = BacktestEngine("lightgbm", 20, 1_000_000.0)
    weekly_df = eng.load_weekly_data()

    df = weekly_df.sort_values(["stock_code", "date"]).copy()

    df["close_return"] = df.groupby("stock_code", sort=False)["close"].pct_change()
    df["expected_return"] = df.groupby("stock_code", sort=False)["future_return_1w"].shift(1)

    valid = df.dropna(subset=["close_return", "expected_return"])

    if valid.empty:
        print("无有效数据")
        return

    diff = valid["close_return"] - valid["expected_return"]
    mean_diff = float(diff.mean())
    max_abs_diff = float(diff.abs().max())
    corr = float(valid["close_return"].corr(valid["expected_return"]))

    print(f"close_return[t] vs future_return_1w[t-1]:")
    print(f"  平均差异: {mean_diff:.10f}")
    print(f"  最大绝对差异: {max_abs_diff:.10f}")
    print(f"  相关系数: {corr:.6f}")

    if max_abs_diff < 1e-6:
        print("✅ 合成close价格完全正确：close[t+1]/close[t]-1 = future_return_1w[t]")
    else:
        print(f"⚠️ 合成close价格有误差！最大差异={max_abs_diff:.10f}")
        worst = valid.nlargest(5, "close_return")
        for _, row in worst.iterrows():
            print(f"  {row['stock_code']} @ {row['date']}: close_ret={row['close_return']:.6f}, fr1w_prev={row['expected_return']:.6f}")


def check_holding_return_calculation() -> None:
    """直接计算：如果按future_return_1w等权持有Top20，理论收益是多少？"""
    print()
    print("=" * 60)
    print("测试4：理论等权持有收益 vs 回测收益")
    print("=" * 60)

    from src.predictor import ModelPredictor

    test_path = Path(__file__).resolve().parent.parent / "data" / "test.parquet"
    df = pd.read_parquet(test_path)

    pred = ModelPredictor("lightgbm", data_dir=Path(__file__).resolve().parent.parent / "data")
    factor_cols = pred._factor_cols

    dates = sorted(df["date"].unique())

    nav_theoretical = 1.0
    nav_points = [1.0]

    for i in range(1, len(dates)):
        prev_date = dates[i - 1]
        curr_date = dates[i]

        prev_section = df[df["date"] == prev_date].copy()
        avail_cols = [c for c in factor_cols if c in prev_section.columns]
        if len(avail_cols) < 5:
            nav_points.append(nav_theoretical)
            continue

        try:
            pred_df = pred.predict(prev_section[["stock_code"] + avail_cols])
        except Exception:
            nav_points.append(nav_theoretical)
            continue

        top20_codes = pred_df.nlargest(20, "score")["stock_code"].tolist()

        curr_section = df[df["date"] == curr_date]
        top20_fr1w = prev_section[prev_section["stock_code"].isin(top20_codes)]["future_return_1w"].dropna()

        if len(top20_fr1w) > 0:
            avg_ret = float(top20_fr1w.mean())
            nav_theoretical *= (1 + avg_ret)
        nav_points.append(nav_theoretical)

    n_weeks = len(dates) - 1
    annualized = (nav_theoretical ** (52 / max(n_weeks, 1))) - 1

    print(f"理论等权持有Top20（无交易成本）:")
    print(f"  最终净值: {nav_theoretical:.4f}")
    print(f"  年化收益: {annualized * 100:.2f}%")

    eng = BacktestEngine("lightgbm", 20, 1_000_000.0, commission=0.0, slippage=0.0, stamp_tax=0.0)
    weekly_df = eng.load_weekly_data()
    result_df = eng.run_backtest(weekly_df, use_split="test")
    if not result_df.empty:
        m = compute_backtest_metrics(result_df, weekly_df, extended=False)
        bt_ann = m.get("annualized_return", 0)
        print(f"零成本回测Top20:")
        print(f"  年化收益: {bt_ann * 100:.2f}%")
        print(f"  差异: {(annualized - bt_ann) * 100:.2f}%")


if __name__ == "__main__":
    run_zero_cost_backtest()
    check_cash_drag()
    check_synthetic_close_correctness()
    check_holding_return_calculation()
