"""
实验1：收益-夏普感知损失函数对比

实验组：
- B-LGBM: 标准LambdaRank NDCG（基线，使用已有模型）
- E1a-LGBM: 收益加权NDCG
- E1b-LGBM: 收益加权NDCG + 夏普惩罚
- E1c-LGBM: 收益加权NDCG + 夏普惩罚 + CVaR

用法：
    cd backend
    python scripts/run_experiment1.py
    python scripts/run_experiment1.py --skip-train  # 跳过训练，只跑回测
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model_lightgbm import (
    return_aware_relevance,
    sharpe_aware_relevance,
    cvar_aware_relevance,
    train_final_lightgbm_with_label_fn,
)
from src.backtest import BacktestEngine, compute_backtest_metrics
from src.predictor import ModelPredictor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
EXPERIMENT_DIR = DATA_DIR / "experiment1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("experiment1")


def e1a_label_fn(
    y: np.ndarray,
    group_sizes: list[int],
    max_label: int = 30,
    alpha: float = 1.0,
    volatility: np.ndarray | None = None,
    **_kwargs: Any,
) -> np.ndarray:
    """
    E1a标签函数包装：收益加权NDCG。

    包装return_aware_relevance，忽略build_datasets_with_label_fn自动注入的volatility参数。

    参数:
        y: 未来收益率数组
        group_sizes: 截面样本数列表
        max_label: 标签上界
        alpha: 收益加权幂次
        volatility: 自动注入的波动率数组，E1a不使用
        **_kwargs: 其他自动注入参数，忽略

    返回:
        整数relevance标签数组
    """
    return return_aware_relevance(y, group_sizes, max_label=max_label, alpha=alpha)


def e1b_label_fn(
    y: np.ndarray,
    group_sizes: list[int],
    max_label: int = 30,
    alpha: float = 1.0,
    volatility: np.ndarray | None = None,
    sharpe_penalty: float = 0.5,
    **_kwargs: Any,
) -> np.ndarray:
    """
    E1b标签函数包装：收益加权NDCG + 夏普惩罚。

    包装sharpe_aware_relevance，兼容自动注入参数。

    参数:
        y: 未来收益率数组
        group_sizes: 截面样本数列表
        max_label: 标签上界
        alpha: 收益加权幂次
        volatility: 自动注入的波动率数组
        sharpe_penalty: 夏普惩罚系数
        **_kwargs: 其他自动注入参数，忽略

    返回:
        整数relevance标签数组
    """
    return sharpe_aware_relevance(
        y, group_sizes,
        max_label=max_label, alpha=alpha,
        volatility=volatility, sharpe_penalty=sharpe_penalty,
    )


def e1c_label_fn(
    y: np.ndarray,
    group_sizes: list[int],
    max_label: int = 30,
    alpha: float = 1.0,
    volatility: np.ndarray | None = None,
    sharpe_penalty: float = 0.5,
    cvar_penalty: float = 1.5,
    cvar_alpha: float = 0.20,
    **_kwargs: Any,
) -> np.ndarray:
    """
    E1c标签函数包装：收益加权NDCG + 夏普惩罚 + CVaR。

    包装cvar_aware_relevance，兼容自动注入参数。

    参数:
        y: 未来收益率数组
        group_sizes: 截面样本数列表
        max_label: 标签上界
        alpha: 收益加权幂次
        volatility: 自动注入的波动率数组
        sharpe_penalty: 夏普惩罚系数
        cvar_penalty: CVaR惩罚系数
        cvar_alpha: CVaR分位数阈值
        **_kwargs: 其他自动注入参数，忽略

    返回:
        整数relevance标签数组
    """
    return cvar_aware_relevance(
        y, group_sizes,
        max_label=max_label, alpha=alpha,
        volatility=volatility, sharpe_penalty=sharpe_penalty,
        cvar_penalty=cvar_penalty, cvar_alpha=cvar_alpha,
    )


def train_e1a(n_trials: int = 20) -> None:
    """训练 E1a 模型：收益加权NDCG。"""
    logger.info("=" * 60)
    logger.info("开始训练 E1a-LGBM（收益加权NDCG）")
    logger.info("=" * 60)
    t0 = time.time()
    bst_e1a, params_e1a = train_final_lightgbm_with_label_fn(
        e1a_label_fn,
        label_fn_name="e1a",
        label_fn_kwargs={"alpha": 1.0},
        tune=True,
        n_trials=n_trials,
    )
    elapsed = time.time() - t0
    logger.info("E1a 训练完成，耗时 %.1f 秒，最优参数: %s", elapsed, params_e1a)


def train_e1b(n_trials: int = 20) -> None:
    """训练 E1b 模型：收益加权NDCG + 夏普惩罚。"""
    logger.info("=" * 60)
    logger.info("开始训练 E1b-LGBM（收益加权NDCG + 夏普惩罚）")
    logger.info("=" * 60)
    t0 = time.time()
    bst_e1b, params_e1b = train_final_lightgbm_with_label_fn(
        e1b_label_fn,
        label_fn_name="e1b",
        label_fn_kwargs={"alpha": 1.0, "sharpe_penalty": 0.5},
        tune=True,
        n_trials=n_trials,
    )
    elapsed = time.time() - t0
    logger.info("E1b 训练完成，耗时 %.1f 秒，最优参数: %s", elapsed, params_e1b)


def train_e1c(n_trials: int = 20) -> None:
    """训练 E1c 模型：收益加权NDCG + 夏普惩罚 + CVaR。"""
    logger.info("=" * 60)
    logger.info("开始训练 E1c-LGBM（收益加权NDCG + 夏普惩罚 + CVaR）")
    logger.info("=" * 60)
    t0 = time.time()
    bst_e1c, params_e1c = train_final_lightgbm_with_label_fn(
        e1c_label_fn,
        label_fn_name="e1c",
        label_fn_kwargs={"alpha": 1.0, "sharpe_penalty": 0.5, "cvar_penalty": 1.5, "cvar_alpha": 0.20},
        tune=True,
        n_trials=n_trials,
        tune_seed=123,
    )
    elapsed = time.time() - t0
    logger.info("E1c 训练完成，耗时 %.1f 秒，最优参数: %s", elapsed, params_e1c)


def run_backtest_for_model(
    model_name: str,
    model_path: Path,
    *,
    top_n: int = 20,
    vol_penalty: float = 1.0,
    use_split: str = "test",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    对指定模型运行回测。

    参数:
        model_name: 模型名称（如 B-LGBM、E1a-LGBM）
        model_path: 模型文件路径
        top_n: 选股数量
        vol_penalty: 波动率惩罚系数
        use_split: 数据切分方式

    返回:
        (回测结果DataFrame, 回测指标字典)
    """
    logger.info("开始回测 %s（model_path=%s, vol_penalty=%.1f）", model_name, model_path, vol_penalty)
    predictor = ModelPredictor("lightgbm", model_path=model_path)
    engine = BacktestEngine(
        "lightgbm",
        top_n=top_n,
        vol_penalty=vol_penalty,
    )
    weekly_df = engine.load_weekly_data()
    result_df = engine.run_backtest(
        weekly_df,
        predictor=predictor,
        use_split=use_split,
    )
    if result_df.empty:
        logger.warning("%s 回测结果为空", model_name)
        return result_df, {}
    metrics = compute_backtest_metrics(result_df, weekly_df, extended=True)
    logger.info(
        "%s 回测完成：年化收益=%.2f%%, 夏普=%.3f, 最大回撤=%.2f%%",
        model_name,
        metrics.get("annualized_return", 0) * 100,
        metrics.get("sharpe_ratio", 0),
        metrics.get("max_drawdown", 0) * 100,
    )
    return result_df, metrics


def load_ndcg_metrics(model_label: str) -> dict[str, float]:
    """
    从模型指标JSON文件读取NDCG值。

    参数:
        model_label: 模型标签（如 lightgbm、e1a、e1b、e1c）

    返回:
        包含 ndcg@5、ndcg@10、ndcg@20 的字典
    """
    if model_label == "lightgbm":
        metrics_path = MODELS_DIR / "lightgbm_metrics.json"
    else:
        metrics_path = MODELS_DIR / f"{model_label}_lightgbm_metrics.json"
    if not metrics_path.exists():
        logger.warning("指标文件不存在: %s", metrics_path)
        return {}
    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    val_ndcg = data.get("val_ndcg", {})
    return {
        "ndcg@5": val_ndcg.get("ndcg@5", float("nan")),
        "ndcg@10": val_ndcg.get("ndcg@10", float("nan")),
        "ndcg@20": val_ndcg.get("ndcg@20", float("nan")),
    }


def generate_report(
    all_metrics: dict[str, dict[str, Any]],
    all_ndcg: dict[str, dict[str, float]],
) -> str:
    """
    生成实验1对比报告。

    参数:
        all_metrics: 各模型的回测指标 {model_name: metrics_dict}
        all_ndcg: 各模型的NDCG指标 {model_name: ndcg_dict}

    返回:
        Markdown格式的报告文本
    """
    lines: list[str] = []
    lines.append("# 实验1：收益-夏普感知损失函数对比报告")
    lines.append("")
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    lines.append("## 1. 实验设计")
    lines.append("")
    lines.append("| 实验组 | 标签构造方式 | 核心思路 |")
    lines.append("|--------|------------|---------|")
    lines.append("| B-LGBM | 标准LambdaRank NDCG | 基线，按收益率排名映射为离散标签 |")
    lines.append("| E1a-LGBM | 收益加权NDCG | 在标准标签基础上，根据实际收益率大小额外提升高收益股票标签值 |")
    lines.append("| E1b-LGBM | 收益加权NDCG + 夏普惩罚 | 在E1a基础上，对高波动股票标签进行惩罚 |")
    lines.append("| E1c-LGBM | 收益加权NDCG + 夏普惩罚 + CVaR | 在E1b基础上，对左尾收益（极端亏损）股票额外惩罚 |")
    lines.append("")

    lines.append("## 2. 排序质量（NDCG）")
    lines.append("")
    lines.append("| 模型 | NDCG@5 | NDCG@10 | NDCG@20 |")
    lines.append("|------|--------|---------|---------|")
    for name in ["B-LGBM", "E1a-LGBM", "E1b-LGBM", "E1c-LGBM"]:
        ndcg = all_ndcg.get(name, {})
        lines.append(
            f"| {name} | {ndcg.get('ndcg@5', float('nan')):.4f} "
            f"| {ndcg.get('ndcg@10', float('nan')):.4f} "
            f"| {ndcg.get('ndcg@20', float('nan')):.4f} |"
        )
    lines.append("")

    lines.append("## 3. 回测指标对比")
    lines.append("")
    lines.append("回测参数：top_n=20, vol_penalty=1.0, use_split=test")
    lines.append("")
    lines.append("| 模型 | 年化收益(%) | 夏普比率 | 最大回撤(%) | 换手率 | 周度胜率(%) |")
    lines.append("|------|------------|---------|------------|--------|------------|")
    for name in ["B-LGBM", "E1a-LGBM", "E1b-LGBM", "E1c-LGBM"]:
        m = all_metrics.get(name, {})
        ann_ret = m.get("annualized_return", float("nan"))
        sharpe = m.get("sharpe_ratio", float("nan"))
        mdd = m.get("max_drawdown", float("nan"))
        turnover = m.get("turnover_rate", None)
        win_rate = m.get("win_rate", None)
        ann_ret_str = f"{ann_ret * 100:.2f}" if not np.isnan(ann_ret) else "N/A"
        sharpe_str = f"{sharpe:.3f}" if not np.isnan(sharpe) else "N/A"
        mdd_str = f"{mdd * 100:.2f}" if not np.isnan(mdd) else "N/A"
        turnover_str = f"{turnover:.2f}" if turnover is not None else "N/A"
        win_rate_str = f"{win_rate * 100:.1f}" if win_rate is not None else "N/A"
        lines.append(f"| {name} | {ann_ret_str} | {sharpe_str} | {mdd_str} | {turnover_str} | {win_rate_str} |")
    lines.append("")

    lines.append("## 4. B-LGBM vs 实验组对比")
    lines.append("")
    baseline_metrics = all_metrics.get("B-LGBM", {})
    baseline_ann_ret = baseline_metrics.get("annualized_return", 0)
    baseline_sharpe = baseline_metrics.get("sharpe_ratio", 0)
    baseline_mdd = baseline_metrics.get("max_drawdown", 0)
    lines.append("| 对比 | 年化收益差(pp) | 夏普差 | 最大回撤差(pp) | 是否优于基线 |")
    lines.append("|------|--------------|--------|--------------|------------|")
    for name in ["E1a-LGBM", "E1b-LGBM", "E1c-LGBM"]:
        m = all_metrics.get(name, {})
        ann_ret = m.get("annualized_return", float("nan"))
        sharpe = m.get("sharpe_ratio", float("nan"))
        mdd = m.get("max_drawdown", float("nan"))
        if np.isnan(ann_ret) or np.isnan(sharpe) or np.isnan(mdd):
            lines.append(f"| {name} vs B-LGBM | N/A | N/A | N/A | N/A |")
            continue
        ret_diff = (ann_ret - baseline_ann_ret) * 100
        sharpe_diff = sharpe - baseline_sharpe
        mdd_diff = (mdd - baseline_mdd) * 100
        better = ann_ret > baseline_ann_ret and sharpe > baseline_sharpe
        lines.append(
            f"| {name} vs B-LGBM | {ret_diff:+.2f} | {sharpe_diff:+.3f} "
            f"| {mdd_diff:+.2f} | {'✅ 是' if better else '❌ 否'} |"
        )
    lines.append("")

    lines.append("## 5. 关键论证逻辑")
    lines.append("")
    lines.append("### 5.1 实验假设")
    lines.append("")
    lines.append("标准LambdaRank仅关注相对排名（NDCG），忽略了收益的绝对大小和风险特征。")
    lines.append("通过在标签构造中引入收益加权和风险惩罚，可以使模型在排序时更关注：")
    lines.append("- **E1a**：绝对收益水平（高收益股票获得更高标签值）")
    lines.append("- **E1b**：收益/风险比（高波动股票标签被惩罚）")
    lines.append("- **E1c**：下行风险规避（极端亏损股票额外惩罚）")
    lines.append("")
    lines.append("### 5.2 预期效果")
    lines.append("")
    lines.append("1. E1a应使模型更倾向选择高收益股票，但可能增加组合波动")
    lines.append("2. E1b在E1a基础上通过波动率惩罚降低组合风险，预期夏普比率提升")
    lines.append("3. E1c在E1b基础上通过CVaR惩罚规避极端亏损，预期最大回撤改善")
    lines.append("")
    lines.append("### 5.3 实验结论")
    lines.append("")

    e1a_better = False
    e1b_better = False
    e1c_better = False
    e1a_metrics = all_metrics.get("E1a-LGBM", {})
    e1b_metrics = all_metrics.get("E1b-LGBM", {})
    e1c_metrics = all_metrics.get("E1c-LGBM", {})
    if e1a_metrics and baseline_metrics:
        e1a_ann = e1a_metrics.get("annualized_return", float("nan"))
        e1a_sharpe = e1a_metrics.get("sharpe_ratio", float("nan"))
        if not np.isnan(e1a_ann) and not np.isnan(e1a_sharpe):
            e1a_better = e1a_ann > baseline_ann_ret and e1a_sharpe > baseline_sharpe
    if e1b_metrics and baseline_metrics:
        e1b_ann = e1b_metrics.get("annualized_return", float("nan"))
        e1b_sharpe = e1b_metrics.get("sharpe_ratio", float("nan"))
        if not np.isnan(e1b_ann) and not np.isnan(e1b_sharpe):
            e1b_better = e1b_ann > baseline_ann_ret and e1b_sharpe > baseline_sharpe
    if e1c_metrics and baseline_metrics:
        e1c_ann = e1c_metrics.get("annualized_return", float("nan"))
        e1c_sharpe = e1c_metrics.get("sharpe_ratio", float("nan"))
        if not np.isnan(e1c_ann) and not np.isnan(e1c_sharpe):
            e1c_better = e1c_ann > baseline_ann_ret and e1c_sharpe > baseline_sharpe

    if e1a_better or e1b_better or e1c_better:
        lines.append("收益-夏普感知损失函数**有效**，至少一组实验优于基线：")
        if e1a_better:
            lines.append("- ✅ E1a（收益加权NDCG）优于B-LGBM")
        else:
            lines.append("- ❌ E1a（收益加权NDCG）未优于B-LGBM")
        if e1b_better:
            lines.append("- ✅ E1b（收益加权NDCG + 夏普惩罚）优于B-LGBM")
        else:
            lines.append("- ❌ E1b（收益加权NDCG + 夏普惩罚）未优于B-LGBM")
        if e1c_better:
            lines.append("- ✅ E1c（收益加权NDCG + 夏普惩罚 + CVaR）优于B-LGBM")
        else:
            lines.append("- ❌ E1c（收益加权NDCG + 夏普惩罚 + CVaR）未优于B-LGBM")
    else:
        lines.append("收益-夏普感知损失函数**未显著优于基线**，可能原因：")
        lines.append("- 标签调整幅度不足以改变模型排序行为")
        lines.append("- vol_penalty已在回测层面实现了风险调整，标签层面的调整边际效果有限")
        lines.append("- A股周频数据中收益与波动的非线性关系使得简单惩罚难以有效建模")
    lines.append("")

    lines.append("## 6. 详细指标")
    lines.append("")
    for name in ["B-LGBM", "E1a-LGBM", "E1b-LGBM", "E1c-LGBM"]:
        m = all_metrics.get(name, {})
        if not m:
            continue
        lines.append(f"### {name}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(m, ensure_ascii=False, indent=2, default=str))
        lines.append("```")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="实验1：收益-夏普感知损失函数对比")
    parser.add_argument("--skip-train", action="store_true", help="跳过训练，只跑回测")
    parser.add_argument("--n-trials", type=int, default=20, help="Optuna搜索次数（默认20）")
    args = parser.parse_args()

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_train:
        logger.info("开始训练实验模型...")
        train_e1a(n_trials=args.n_trials)
        train_e1b(n_trials=args.n_trials)
        train_e1c(n_trials=args.n_trials)
        logger.info("所有模型训练完成")
    else:
        logger.info("跳过训练，直接使用已有模型")

    model_configs: list[dict[str, Any]] = [
        {"name": "B-LGBM", "path": MODELS_DIR / "lightgbm.pkl", "label": "lightgbm"},
        {"name": "E1a-LGBM", "path": MODELS_DIR / "e1a_lightgbm.pkl", "label": "e1a"},
        {"name": "E1b-LGBM", "path": MODELS_DIR / "e1b_lightgbm.pkl", "label": "e1b"},
        {"name": "E1c-LGBM", "path": MODELS_DIR / "e1c_lightgbm.pkl", "label": "e1c"},
    ]

    all_metrics: dict[str, dict[str, Any]] = {}
    all_ndcg: dict[str, dict[str, float]] = {}

    for cfg in model_configs:
        model_name = cfg["name"]
        model_path = cfg["path"]
        model_label = cfg["label"]

        if not model_path.exists():
            logger.error("模型文件不存在: %s，跳过 %s", model_path, model_name)
            all_metrics[model_name] = {}
            all_ndcg[model_name] = {}
            continue

        ndcg = load_ndcg_metrics(model_label)
        all_ndcg[model_name] = ndcg

        result_df, metrics = run_backtest_for_model(
            model_name,
            model_path,
            top_n=20,
            vol_penalty=1.0,
            use_split="test",
        )
        all_metrics[model_name] = metrics

        result_path = EXPERIMENT_DIR / f"{model_label}_backtest_result.parquet"
        if not result_df.empty:
            result_df.to_parquet(result_path, index=False)
            logger.info("回测结果已保存: %s", result_path)

        metrics_path = EXPERIMENT_DIR / f"{model_label}_backtest_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2, default=str)
        logger.info("回测指标已保存: %s", metrics_path)

    report = generate_report(all_metrics, all_ndcg)
    report_path = EXPERIMENT_DIR / "experiment1_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info("实验报告已保存: %s", report_path)

    summary_path = EXPERIMENT_DIR / "experiment1_summary.json"
    summary: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "models": {},
    }
    for name in ["B-LGBM", "E1a-LGBM", "E1b-LGBM", "E1c-LGBM"]:
        m = all_metrics.get(name, {})
        ndcg = all_ndcg.get(name, {})
        summary["models"][name] = {
            "ndcg@10": ndcg.get("ndcg@10"),
            "annualized_return": m.get("annualized_return"),
            "sharpe_ratio": m.get("sharpe_ratio"),
            "max_drawdown": m.get("max_drawdown"),
            "turnover_rate": m.get("turnover_rate"),
            "win_rate": m.get("win_rate"),
        }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    logger.info("摘要已保存: %s", summary_path)

    logger.info("=" * 60)
    logger.info("实验1完成！核心结果摘要：")
    logger.info("=" * 60)
    for name in ["B-LGBM", "E1a-LGBM", "E1b-LGBM", "E1c-LGBM"]:
        m = all_metrics.get(name, {})
        ndcg = all_ndcg.get(name, {})
        ann_ret = m.get("annualized_return", float("nan"))
        sharpe = m.get("sharpe_ratio", float("nan"))
        mdd = m.get("max_drawdown", float("nan"))
        ndcg10 = ndcg.get("ndcg@10", float("nan"))
        logger.info(
            "%s: NDCG@10=%.4f, 年化收益=%.2f%%, 夏普=%.3f, 最大回撤=%.2f%%",
            name,
            ndcg10,
            ann_ret * 100 if not np.isnan(ann_ret) else float("nan"),
            sharpe,
            mdd * 100 if not np.isnan(mdd) else float("nan"),
        )


if __name__ == "__main__":
    main()
