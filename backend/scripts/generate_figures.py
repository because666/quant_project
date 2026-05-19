"""
论文图表生成脚本

从实验结果JSON中读取数据，生成8张论文图表。
输出到 data/figures/ 目录，DPI=300，中文SimHei字体。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FIGURES_DIR = PROJECT_ROOT / "data" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = PROJECT_ROOT / "data"

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"

PALETTE = sns.color_palette("Set2")
METHOD_COLORS = {
    "M1-B-LGBM": PALETTE[0],
    "M2-B-XGB": PALETTE[1],
    "M3-E1b-RRF": PALETTE[2],
    "M4-E1a-AVG": PALETTE[3],
}
METHOD_LABELS = {
    "M1-B-LGBM": "M1: LightGBM基线",
    "M2-B-XGB": "M2: XGBoost基线",
    "M3-E1b-RRF": "M3: 收益-夏普+RRF",
    "M4-E1a-AVG": "M4: 收益加权+平均",
}
ALL_METHOD_LABELS = {
    "B-LGBM": "LightGBM基线",
    "B-XGB": "XGBoost基线",
    "E1a-LGBM": "收益加权NDCG",
    "E1b-LGBM": "收益+夏普感知",
    "E2a-AVG": "分数平均融合",
    "E2b-RRF": "RRF融合",
}


def load_json(path: Path) -> Any:
    """加载JSON文件。"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fig1_comprehensive_comparison() -> None:
    """图1：综合对比分组柱状图。"""
    data = load_json(DATA_DIR / "experiment4" / "experiment4_summary.json")
    methods_data = data["methods"]

    method_order = ["M1-B-LGBM", "M2-B-XGB", "M3-E1b-RRF", "M4-E1a-AVG"]
    labels = [METHOD_LABELS[m] for m in method_order]

    ann_ret = [methods_data[m]["annualized_return"] * 100 for m in method_order]
    sharpe = [methods_data[m]["sharpe_ratio"] for m in method_order]
    max_dd = [methods_data[m]["max_drawdown"] * 100 for m in method_order]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    x = np.arange(len(labels))
    w = 0.6

    bars = axes[0].bar(x, ann_ret, w, color=[METHOD_COLORS[m] for m in method_order])
    axes[0].set_ylabel("年化收益率 (%)")
    axes[0].set_title("年化收益率")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    for bar, val in zip(bars, ann_ret):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     f"{val:.1f}%", ha="center", va="bottom", fontsize=8)

    bars = axes[1].bar(x, sharpe, w, color=[METHOD_COLORS[m] for m in method_order])
    axes[1].set_ylabel("夏普比率")
    axes[1].set_title("夏普比率")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    for bar, val in zip(bars, sharpe):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    bars = axes[2].bar(x, max_dd, w, color=[METHOD_COLORS[m] for m in method_order])
    axes[2].set_ylabel("最大回撤 (%)")
    axes[2].set_title("最大回撤")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    for bar, val in zip(bars, max_dd):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                     f"{val:.1f}%", ha="center", va="bottom", fontsize=8)

    fig.suptitle("图1：综合对比", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_comprehensive_comparison.png")
    plt.close(fig)
    print("图1 已生成")


def fig2_quantile_monotonicity() -> None:
    """图2：分位数单调性折线图。"""
    data = load_json(DATA_DIR / "experiment4" / "quantile_analysis.json")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    quantile_labels = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    x = np.arange(len(quantile_labels))

    for method_key, color in METHOD_COLORS.items():
        if method_key not in data:
            continue
        ann_rets = [data[method_key][q]["annualized_return"] * 100 for q in quantile_labels]
        sharpes = [data[method_key][q]["sharpe_ratio"] for q in quantile_labels]

        axes[0].plot(x, ann_rets, "o-", color=color, label=METHOD_LABELS[method_key], linewidth=2)
        axes[1].plot(x, sharpes, "o-", color=color, label=METHOD_LABELS[method_key], linewidth=2)

    for ax, title, ylabel in [
        (axes[0], "分位数组合年化收益率", "年化收益率 (%)"),
        (axes[1], "分位数组合夏普比率", "夏普比率"),
    ]:
        ax.set_xticks(x)
        ax.set_xticklabels(quantile_labels)
        ax.set_xlabel("分位数分组")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    ls_text = "多空(LS)年化收益: " + ", ".join(
        f"{METHOD_LABELS[m]}={data[m]['LS']['annualized_return']*100:.1f}%"
        for m in METHOD_COLORS if m in data
    )
    fig.text(0.5, -0.02, ls_text, ha="center", fontsize=9, style="italic")

    fig.suptitle("图2：分位数组合单调性验证", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_quantile_monotonicity.png")
    plt.close(fig)
    print("图2 已生成")


def fig3_yearly_comparison() -> None:
    """图3：分年度对比柱状图。"""
    data = load_json(DATA_DIR / "experiment4" / "yearly_analysis.json")

    method_order = ["M1-B-LGBM", "M2-B-XGB", "M3-E1b-RRF", "M4-E1a-AVG"]
    years = sorted(set(yr for m in method_order for yr in data.get(m, {})))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(method_order))
    w = 0.35

    for i, year in enumerate(years):
        offset = (i - len(years) / 2 + 0.5) * w
        ann_rets = [data.get(m, {}).get(year, {}).get("annualized_return", 0) * 100 for m in method_order]
        sharpes = [data.get(m, {}).get(year, {}).get("sharpe_ratio", 0) for m in method_order]

        axes[0].bar(x + offset, ann_rets, w, label=f"{year}年")
        axes[1].bar(x + offset, sharpes, w, label=f"{year}年")

    labels = [METHOD_LABELS[m] for m in method_order]
    for ax, title, ylabel in [
        (axes[0], "分年度年化收益率", "年化收益率 (%)"),
        (axes[1], "分年度夏普比率", "夏普比率"),
    ]:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("图3：分年度对比", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_yearly_comparison.png")
    plt.close(fig)
    print("图3 已生成")


def fig4_bootstrap_ci() -> None:
    """图4：Bootstrap CI误差棒图。"""
    data = load_json(DATA_DIR / "statistical_tests" / "bootstrap_ci.json")

    method_order = ["B-LGBM", "B-XGB", "E1a-LGBM", "E1b-LGBM", "E2a-AVG", "E2b-RRF"]
    labels = [ALL_METHOD_LABELS[m] for m in method_order]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, metric, title, ylabel in [
        (0, "annualized_return", "年化收益率 Bootstrap 95% CI", "年化收益率 (%)"),
        (1, "sharpe_ratio", "夏普比率 Bootstrap 95% CI", "夏普比率"),
    ]:
        points = []
        lowers = []
        uppers = []
        for m in method_order:
            ci = data.get(m, {}).get(metric, {})
            pt = ci.get("point", 0)
            lo = ci.get("lower", 0)
            up = ci.get("upper", 0)
            if metric == "annualized_return":
                pt, lo, up = pt * 100, lo * 100, up * 100
            points.append(pt)
            lowers.append(lo)
            uppers.append(up)

        x = np.arange(len(labels))
        yerr_low = [p - l for p, l in zip(points, lowers)]
        yerr_up = [u - p for p, u in zip(points, uppers)]

        colors = [PALETTE[i % len(PALETTE)] for i in range(len(method_order))]
        axes[ax_idx].bar(x, points, 0.6, color=colors, yerr=[yerr_low, yerr_up],
                        capsize=5, error_kw={"linewidth": 1.5})
        axes[ax_idx].set_xticks(x)
        axes[ax_idx].set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        axes[ax_idx].set_ylabel(ylabel)
        axes[ax_idx].set_title(title)
        axes[ax_idx].axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
        axes[ax_idx].grid(True, alpha=0.3, axis="y")

    fig.suptitle("图4：Bootstrap 95%置信区间", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig4_bootstrap_ci.png")
    plt.close(fig)
    print("图4 已生成")


def fig5_topn_sensitivity() -> None:
    """图5：Top N敏感性折线图。"""
    data = load_json(DATA_DIR / "statistical_tests" / "sensitivity_topn.json")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    method_colors = {
        "B-LGBM": PALETTE[0],
        "B-XGB": PALETTE[1],
        "E1a-LGBM": PALETTE[2],
        "E1b-LGBM": PALETTE[3],
    }

    for method, color in method_colors.items():
        if method not in data:
            continue
        topn_data = data[method]
        topns = sorted(int(k) for k in topn_data.keys())
        ann_rets = [topn_data[str(t)]["annualized_return"] * 100 for t in topns]
        sharpes = [topn_data[str(t)]["sharpe_ratio"] for t in topns]

        axes[0].plot(topns, ann_rets, "o-", color=color, label=ALL_METHOD_LABELS[method], linewidth=2)
        axes[1].plot(topns, sharpes, "o-", color=color, label=ALL_METHOD_LABELS[method], linewidth=2)

    for ax, title, ylabel in [
        (axes[0], "Top N参数敏感性 — 年化收益率", "年化收益率 (%)"),
        (axes[1], "Top N参数敏感性 — 夏普比率", "夏普比率"),
    ]:
        ax.set_xlabel("Top N")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("图5：Top N参数敏感性分析", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig5_topn_sensitivity.png")
    plt.close(fig)
    print("图5 已生成")


def fig6_holding_sensitivity() -> None:
    """图6：持有期敏感性折线图。"""
    data = load_json(DATA_DIR / "statistical_tests" / "sensitivity_holding.json")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    method_colors = {
        "B-LGBM": PALETTE[0],
        "B-XGB": PALETTE[1],
        "E1a-LGBM": PALETTE[2],
        "E1b-LGBM": PALETTE[3],
    }

    for method, color in method_colors.items():
        if method not in data:
            continue
        holding_data = data[method]
        periods = sorted(int(k) for k in holding_data.keys())
        ann_rets = [holding_data[str(p)]["annualized_return"] * 100 for p in periods]
        sharpes = [holding_data[str(p)]["sharpe_ratio"] for p in periods]

        axes[0].plot(periods, ann_rets, "o-", color=color, label=ALL_METHOD_LABELS[method], linewidth=2)
        axes[1].plot(periods, sharpes, "o-", color=color, label=ALL_METHOD_LABELS[method], linewidth=2)

    for ax, title, ylabel in [
        (axes[0], "持有期敏感性 — 年化收益率", "年化收益率 (%)"),
        (axes[1], "持有期敏感性 — 夏普比率", "夏普比率"),
    ]:
        ax.set_xlabel("持有期（周）")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks([1, 2, 4])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("图6：持有期敏感性分析", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig6_holding_sensitivity.png")
    plt.close(fig)
    print("图6 已生成")


def fig7_paired_test_heatmap() -> None:
    """图7：配对检验热力图。"""
    data = load_json(DATA_DIR / "statistical_tests" / "paired_tests.json")

    method_order = ["B-LGBM", "B-XGB", "E1a-LGBM", "E1b-LGBM", "E2a-AVG", "E2b-RRF"]
    n = len(method_order)
    p_matrix = np.ones((n, n))
    sig_matrix = np.zeros((n, n), dtype=bool)

    for test in data:
        a = test["method_a"]
        b = test["method_b"]
        if a in method_order and b in method_order:
            i = method_order.index(a)
            j = method_order.index(b)
            p_val = test["p_value"]
            p_matrix[i, j] = p_val
            p_matrix[j, i] = p_val
            sig_matrix[i, j] = test.get("significant", False)
            sig_matrix[j, i] = test.get("significant", False)

    fig, ax = plt.subplots(figsize=(8, 7))

    mask = np.triu(np.ones_like(p_matrix, dtype=bool), k=1)
    sns.heatmap(p_matrix, mask=mask, annot=True, fmt=".3f", cmap="RdYlGn_r",
                xticklabels=[ALL_METHOD_LABELS[m] for m in method_order],
                yticklabels=[ALL_METHOD_LABELS[m] for m in method_order],
                ax=ax, vmin=0, vmax=1, linewidths=0.5,
                cbar_kws={"label": "p值"})

    for i in range(n):
        for j in range(i):
            if sig_matrix[i, j]:
                ax.text(j + 0.5, i + 0.65, "*", ha="center", va="center",
                        fontsize=14, fontweight="bold", color="black")

    ax.set_title("图7：配对显著性检验 p值矩阵\n（下三角，* p<0.05）", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig7_paired_test_heatmap.png")
    plt.close(fig)
    print("图7 已生成")


def fig8_dsr_comparison() -> None:
    """图8：DSR对比柱状图。"""
    data = load_json(DATA_DIR / "statistical_tests" / "deflated_sharpe.json")

    method_order = ["B-LGBM", "B-XGB", "E1a-LGBM", "E1b-LGBM", "E2a-AVG", "E2b-RRF"]
    labels = [ALL_METHOD_LABELS[m] for m in method_order]

    dsr_values = [data.get(m, {}).get("dsr", 0) for m in method_order]
    sharpe_values = [data.get(m, {}).get("sharpe_ratio", 0) for m in method_order]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(labels))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(method_order))]

    bars = ax.bar(x, dsr_values, 0.6, color=colors, alpha=0.85)
    ax.axhline(y=0.95, color="red", linestyle="--", linewidth=2, label="DSR=0.95阈值")

    for bar, dsr, sr in zip(bars, dsr_values, sharpe_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"DSR={dsr:.3f}\nSR={sr:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Deflated Sharpe Ratio")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    ax.set_title("图8：Deflated Sharpe Ratio对比\n（校正20次超参搜索的多重试验偏差）",
                 fontsize=13, fontweight="bold")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig8_dsr_comparison.png")
    plt.close(fig)
    print("图8 已生成")


def main() -> None:
    """生成所有论文图表。"""
    print("=" * 50)
    print("论文图表生成")
    print("=" * 50)

    fig1_comprehensive_comparison()
    fig2_quantile_monotonicity()
    fig3_yearly_comparison()
    fig4_bootstrap_ci()
    fig5_topn_sensitivity()
    fig6_holding_sensitivity()
    fig7_paired_test_heatmap()
    fig8_dsr_comparison()

    print("=" * 50)
    print(f"全部8张图表已生成到: {FIGURES_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    main()
