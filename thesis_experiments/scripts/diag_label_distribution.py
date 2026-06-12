"""
E1b/E1c 标签分布诊断脚本。

分析 4 个标签函数（future_return_to_relevance / return_aware_relevance /
sharpe_aware_relevance / cvar_aware_relevance）在训练集上的标签分布，
特别关注 E1b/E1c 标签是否被过度压缩到 0。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_training_data
from src.model_lightgbm import (
    cvar_aware_relevance,
    future_return_to_relevance,
    return_aware_relevance,
    sharpe_aware_relevance,
)


def _compute_stats(labels: np.ndarray, name: str) -> dict[str, object]:
    """
    计算标签分布统计指标。

    参数:
        labels: 标签数组，形状 (N,)
        name: 标签函数名称

    返回:
        包含均值、标准差、零值比例、非零比例、分位数、变异系数的字典
    """
    n = len(labels)
    mean_val = float(np.mean(labels))
    std_val = float(np.std(labels))
    zero_count = int(np.sum(labels == 0))
    nonzero_count = n - zero_count
    zero_ratio = zero_count / n if n > 0 else 0.0
    nonzero_ratio = nonzero_count / n if n > 0 else 0.0
    cv = std_val / mean_val if abs(mean_val) > 1e-12 else float("inf")
    quantiles = np.percentile(labels, [0, 25, 50, 75, 100])
    return {
        "name": name,
        "n": n,
        "mean": mean_val,
        "std": std_val,
        "zero_count": zero_count,
        "zero_ratio": zero_ratio,
        "nonzero_count": nonzero_count,
        "nonzero_ratio": nonzero_ratio,
        "q0": float(quantiles[0]),
        "q25": float(quantiles[1]),
        "q50": float(quantiles[2]),
        "q75": float(quantiles[3]),
        "q100": float(quantiles[4]),
        "cv": cv,
    }


def _print_stats(stats: dict[str, object]) -> None:
    """格式化输出单个标签函数的统计结果。"""
    print(f"\n{'=' * 60}")
    print(f"标签函数: {stats['name']}")
    print(f"{'=' * 60}")
    print(f"  样本数:        {stats['n']}")
    print(f"  均值:          {stats['mean']:.4f}")
    print(f"  标准差:        {stats['std']:.4f}")
    print(f"  零值数量:      {stats['zero_count']}")
    print(f"  零值比例:      {stats['zero_ratio']:.4%}")
    print(f"  非零数量:      {stats['nonzero_count']}")
    print(f"  非零比例:      {stats['nonzero_ratio']:.4%}")
    print(f"  分位数 (0%):   {stats['q0']:.2f}")
    print(f"  分位数 (25%):  {stats['q25']:.2f}")
    print(f"  分位数 (50%):  {stats['q50']:.2f}")
    print(f"  分位数 (75%):  {stats['q75']:.2f}")
    print(f"  分位数 (100%): {stats['q100']:.2f}")
    print(f"  变异系数(CV):  {stats['cv']:.4f}")


def _print_histogram(labels: np.ndarray, name: str, bins: int = 31) -> None:
    """
    输出标签值分布直方图（0-30 区间，每个整数一个桶）。

    参数:
        labels: 标签数组
        name: 标签函数名称
        bins: 直方图桶数（0~30 共 31 个桶）
    """
    print(f"\n  [{name}] 标签值分布直方图 (0-30):")
    hist, edges = np.histogram(labels, bins=bins, range=(0, 30))
    max_count = max(hist) if max(hist) > 0 else 1
    bar_width = 40
    for i in range(len(hist)):
        bar_len = int(hist[i] / max_count * bar_width)
        bar = "#" * bar_len
        label_val = int(edges[i])
        print(f"    {label_val:3d} | {hist[i]:>8d} {bar}")
    print(f"    总计: {len(labels)}")


def _print_comparison(all_stats: list[dict[str, object]]) -> None:
    """输出 4 个标签函数的对比汇总表。"""
    print(f"\n{'=' * 80}")
    print("对比汇总表")
    print(f"{'=' * 80}")
    header = f"{'指标':<18}"
    for s in all_stats:
        header += f"{s['name']:>16}"
    print(header)
    print("-" * 80)

    metrics = [
        ("均值", "mean", ".4f"),
        ("标准差", "std", ".4f"),
        ("零值比例", "zero_ratio", ".4%"),
        ("非零比例", "nonzero_ratio", ".4%"),
        ("中位数(Q50)", "q50", ".2f"),
        ("Q75", "q75", ".2f"),
        ("Q100(最大)", "q100", ".2f"),
        ("变异系数(CV)", "cv", ".4f"),
    ]
    for label, key, fmt in metrics:
        row = f"{label:<18}"
        for s in all_stats:
            row += f"{s[key]:>{16}{fmt}}"
        print(row)


def main() -> None:
    """主函数：加载数据、生成标签、输出统计。"""
    print("加载训练数据...")
    X_train, y_train, groups_train = load_training_data(fill_missing=True)
    print(f"训练集样本数: {len(X_train)}, 因子列数: {len(X_train.columns)}")

    volatility: np.ndarray | None = None
    if "volatility_12w" in X_train.columns:
        volatility = pd.to_numeric(X_train["volatility_12w"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        volatility = np.nan_to_num(volatility, nan=0.0)
        print(f"volatility_12w 已加载, 范围: [{volatility.min():.4f}, {volatility.max():.4f}]")
    else:
        print("警告: volatility_12w 列不存在, E1b/E1c 将退化为 E1a")

    label_configs: list[tuple[str, object, dict[str, object]]] = [
        ("B-LGBM (future_return_to_relevance)", future_return_to_relevance, {}),
        ("E1a (return_aware_relevance)", return_aware_relevance, {}),
        (
            "E1b (sharpe_aware_relevance)",
            sharpe_aware_relevance,
            {"sharpe_penalty": 0.5, "volatility": volatility},
        ),
        (
            "E1c (cvar_aware_relevance)",
            cvar_aware_relevance,
            {
                "sharpe_penalty": 0.5,
                "cvar_penalty": 1.5,
                "cvar_alpha": 0.20,
                "volatility": volatility,
            },
        ),
    ]

    all_stats: list[dict[str, object]] = []
    all_labels: list[tuple[np.ndarray, str]] = []

    for name, fn, kwargs in label_configs:
        print(f"\n生成标签: {name}...")
        labels = fn(y_train, groups_train, **kwargs)
        stats = _compute_stats(labels, name)
        all_stats.append(stats)
        all_labels.append((labels, name))
        _print_stats(stats)
        _print_histogram(labels, name)

    _print_comparison(all_stats)

    print(f"\n{'=' * 80}")
    print("E1b/E1c 过度压缩诊断")
    print(f"{'=' * 80}")
    for stats in all_stats:
        if "E1b" in stats["name"] or "E1c" in stats["name"]:
            zero_ratio = stats["zero_ratio"]
            mean_val = stats["mean"]
            q50 = stats["q50"]
            if zero_ratio > 0.3:
                print(
                    f"  ⚠ {stats['name']}: 零值比例 {zero_ratio:.4%} 过高 (>30%), "
                    f"均值={mean_val:.4f}, 中位数={q50:.2f}"
                )
            elif zero_ratio > 0.15:
                print(
                    f"  ⚡ {stats['name']}: 零值比例 {zero_ratio:.4%} 偏高 (>15%), "
                    f"均值={mean_val:.4f}, 中位数={q50:.2f}"
                )
            else:
                print(
                    f"  ✓ {stats['name']}: 零值比例 {zero_ratio:.4%} 正常, "
                    f"均值={mean_val:.4f}, 中位数={q50:.2f}"
                )

    baseline_zero = all_stats[0]["zero_ratio"]
    for stats in all_stats[1:]:
        delta = stats["zero_ratio"] - baseline_zero
        print(
            f"  {stats['name']} vs 基线: 零值比例差异 = {delta:+.4%} "
            f"({stats['zero_ratio']:.4%} vs {baseline_zero:.4%})"
        )


if __name__ == "__main__":
    main()
