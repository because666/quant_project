"""
验证回测收益与future_return_1w的一致性。

核心问题：诊断显示Top20周均future_return_1w=0.74%，但回测年化=-23.69%。
这两个数据不应该矛盾，需要定位原因。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.backtest import BacktestEngine, compute_backtest_metrics
from src.data_loader import DATA_OUT_DIR, split_by_time
from src.predictor import ModelPredictor


def verify_close_return_consistency() -> None:
    """验证1：合成close价格计算的收益率是否等于future_return_1w。"""
    print("=" * 60)
    print("验证1：合成close价格收益率 vs future_return_1w")
    print("=" * 60)

    eng = BacktestEngine("lightgbm", 20, 1_000_000.0)
    weekly_df = eng.load_weekly_data()

    _, _, test_part = split_by_time(weekly_df, train_end="2020-12-31", val_end="2022-12-31")
    test_part = test_part.sort_values(["stock_code", "date"]).reset_index(drop=True)

    test_part["close_return"] = test_part.groupby("stock_code", sort=False)["close"].pct_change()
    test_part["return_diff"] = test_part["close_return"] - test_part["future_return_1w"]

    valid = test_part.dropna(subset=["close_return", "future_return_1w"])
    if valid.empty:
        print("无有效数据")
        return

    mean_diff = float(valid["return_diff"].mean())
    max_diff = float(valid["return_diff"].abs().max())
    corr = float(valid["close_return"].corr(valid["future_return_1w"]))

    print(f"close_return vs future_return_1w:")
    print(f"  平均差异: {mean_diff:.8f}")
    print(f"  最大绝对差异: {max_diff:.8f}")
    print(f"  相关系数: {corr:.6f}")

    if max_diff < 1e-6:
        print("  ✅ 合成close价格收益率与future_return_1w完全一致")
    else:
        print(f"  ⚠️ 存在差异！最大差异={max_diff:.6f}")
        worst = valid.nlargest(5, "return_diff_abs" if "return_diff_abs" in valid.columns else "return_diff")
        print(f"  差异最大的5条记录:")
        for _, row in worst.iterrows():
            print(f"    {row['stock_code']} @ {row['date']}: close_ret={row['close_return']:.6f}, fr1w={row['future_return_1w']:.6f}, diff={row['return_diff']:.6f}")


def verify_portfolio_return_vs_fr1w() -> None:
    """验证2：回测组合收益是否等于持仓股票的future_return_1w均值。"""
    print()
    print("=" * 60)
    print("验证2：回测组合收益 vs 持仓股票future_return_1w均值")
    print("=" * 60)

    eng = BacktestEngine("lightgbm", 20, 1_000_000.0)
    weekly_df = eng.load_weekly_data()
    result_df = eng.run_backtest(weekly_df, use_split="test")

    if result_df.empty:
        print("回测结果为空")
        return

    result_df["portfolio_return"] = result_df["total_value"].pct_change()

    import json
    fr1w_means: list[float] = []
    port_rets: list[float] = []

    for i in range(1, len(result_df)):
        prev_row = result_df.iloc[i - 1]
        curr_row = result_df.iloc[i]

        holdings_str = prev_row.get("holdings", "{}")
        if isinstance(holdings_str, str):
            try:
                holdings = json.loads(holdings_str)
            except json.JSONDecodeError:
                holdings = {}
        elif isinstance(holdings_str, dict):
            holdings = holdings_str
        else:
            holdings = {}

        if not holdings:
            continue

        curr_date = curr_row["date"]
        section = weekly_df[weekly_df["date"] == curr_date]

        held_codes = list(holdings.keys())
        held_fr1w = section[section["stock_code"].isin(held_codes)]["future_return_1w"].dropna()

        if len(held_fr1w) > 0:
            fr1w_means.append(float(held_fr1w.mean()))
            port_rets.append(float(curr_row["portfolio_return"]))

    if not fr1w_means:
        print("无有效持仓数据")
        return

    avg_fr1w = float(np.mean(fr1w_means))
    avg_port = float(np.mean(port_rets))
    diff = avg_port - avg_fr1w

    print(f"持仓股票平均future_return_1w: {avg_fr1w:.6f} ({avg_fr1w * 100:.4f}%)")
    print(f"回测组合平均周收益: {avg_port:.6f} ({avg_port * 100:.4f}%)")
    print(f"差异: {diff:.6f} ({diff * 100:.4f}%)")

    if abs(diff) < 0.005:
        print("✅ 回测组合收益与持仓future_return_1w基本一致（差异<0.5%）")
    else:
        print(f"⚠️ 差异较大！回测收益与future_return_1w不一致")
        print(f"  可能原因：交易成本、滑点、现金拖累、涨跌停限制等")

    total_cost_estimate = 0.0003 * 2 + 0.001 * 2 + 0.0005
    print(f"  单次调仓成本估算: {total_cost_estimate * 100:.2f}% (佣金万三双边+滑点0.1%双边+印花税万五)")


def verify_ew_baseline_return() -> None:
    """验证3：等权随机选股的收益是否与全市场平均future_return_1w一致。"""
    print()
    print("=" * 60)
    print("验证3：全市场平均future_return_1w vs B-EW基线")
    print("=" * 60)

    test_path = DATA_OUT_DIR / "test.parquet"
    df = pd.read_parquet(test_path)

    avg_fr1w = float(df["future_return_1w"].mean())
    median_fr1w = float(df["future_return_1w"].median())
    annualized_fr1w = avg_fr1w * 52

    print(f"全市场平均future_return_1w: {avg_fr1w:.6f} ({avg_fr1w * 100:.4f}%)")
    print(f"全市场中位数future_return_1w: {median_fr1w:.6f} ({median_fr1w * 100:.4f}%)")
    print(f"年化（简单）: {annualized_fr1w * 100:.2f}%")

    dates = sorted(df["date"].unique())
    date_avg_fr1w = df.groupby("date")["future_return_1w"].mean()
    n_positive = int((date_avg_fr1w > 0).sum())
    print(f"正收益截面数: {n_positive}/{len(dates)} ({n_positive / len(dates) * 100:.1f}%)")

    print()
    print("如果B-EW基线年化=-5.36%，但全市场平均future_return_1w年化=+11.9%，")
    print("说明B-EW回测逻辑可能有问题，或者交易成本拖累严重。")


def verify_backtest_simple() -> None:
    """验证4：最简单的回测验证——买入持有全市场等权组合。"""
    print()
    print("=" * 60)
    print("验证4：全市场等权买入持有收益")
    print("=" * 60)

    test_path = DATA_OUT_DIR / "test.parquet"
    df = pd.read_parquet(test_path)

    dates = sorted(df["date"].unique())

    date_avg_fr1w = df.groupby("date")["future_return_1w"].mean()

    nav = 1.0
    nav_points = [1.0]
    for d in dates[1:]:
        avg_ret = date_avg_fr1w.get(d, 0)
        if not np.isnan(avg_ret):
            nav *= (1 + avg_ret)
        nav_points.append(nav)

    final_nav = nav_points[-1]
    n_weeks = len(dates) - 1
    annualized = (final_nav ** (52 / max(n_weeks, 1))) - 1

    print(f"初始净值: 1.0")
    print(f"最终净值: {final_nav:.4f}")
    print(f"总周数: {n_weeks}")
    print(f"年化收益（等权持有全市场）: {annualized * 100:.2f}%")

    if annualized > 0:
        print("✅ 全市场等权持有收益为正，说明测试期市场并非全面下跌")
        print("   B-EW基线为负的原因可能是回测逻辑中的交易成本或实现问题")
    else:
        print("⚠️ 全市场等权持有收益为负，说明测试期市场确实偏弱")


if __name__ == "__main__":
    verify_close_return_consistency()
    verify_portfolio_return_vs_fr1w()
    verify_ew_baseline_return()
    verify_backtest_simple()
