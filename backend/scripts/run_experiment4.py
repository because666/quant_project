"""
实验4：综合对比 + 分年度分析 + 多空组合 + 统计检验

根据实验方法论4.5节，本脚本执行以下任务：
1. 综合对比表：汇总所有实验的最优方法（M1~M4）
2. 分年度分析：按自然年拆分测试集，计算每年的年化收益和夏普比率
3. 多空组合收益：按模型得分分位数分组，计算Q1-Q5和多空收益
4. 统计检验：Bootstrap置信区间 + 配对显著性检验

输出：
- data/experiment4/experiment4_report.md
- data/experiment4/experiment4_summary.json
- data/experiment4/yearly_analysis.json
- data/experiment4/quantile_analysis.json
- data/experiment4/significance_tests.json
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest import BacktestEngine, compute_backtest_metrics
from src.metrics import (
    annualized_return,
    bootstrap_metric,
    max_drawdown,
    metrics_by_year,
    paired_significance_test,
    quantile_portfolio_returns,
    sharpe_ratio,
)
from src.predictor import ModelPredictor
from src.fusion import FusionPredictor, score_average_fusion, reciprocal_rank_fusion

EXPERIMENT_DIR = PROJECT_ROOT / "data" / "experiment4"
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(EXPERIMENT_DIR / "experiment4.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

TOP_N = 20
INITIAL_CAPITAL = 1_000_000.0
N_QUANTILES = 5
N_BOOTSTRAP = 5000


def load_weekly_test_data() -> pd.DataFrame:
    """
    加载周频测试集数据。

    返回:
        包含 date, stock_code, 因子列, future_return_1w 的DataFrame

    异常:
        若数据文件不存在或格式异常，抛出FileNotFoundError。
    """
    engine = BacktestEngine(model_type="lightgbm", top_n=TOP_N, initial_capital=INITIAL_CAPITAL)
    weekly_df = engine.load_weekly_data(concat_splits=True)
    test_df = weekly_df[weekly_df["date"] >= "2022-07-01"].copy()
    test_df = test_df.sort_values(["date", "stock_code"]).reset_index(drop=True)
    logger.info("测试集加载完成: %d行, %d个截面", len(test_df), test_df["date"].nunique())
    return test_df


def get_method_configurations() -> dict[str, dict[str, Any]]:
    """
    定义实验4的4种方法配置。

    M1: LightGBM基线（标准NDCG）
    M2: XGBoost基线（标准NDCG）
    M3: 收益-夏普感知损失 + RRF融合
    M4: 收益-夏普感知损失 + RRF融合（最佳损失函数变体）

    返回:
        方法名到配置字典的映射
    """
    return {
        "M1-B-LGBM": {
            "label": "M1: LightGBM基线",
            "predictor_type": "single",
            "model_type": "lightgbm",
            "model_path": MODELS_DIR / "lightgbm.pkl",
            "vol_penalty": 1.0,
        },
        "M2-B-XGB": {
            "label": "M2: XGBoost基线",
            "predictor_type": "single",
            "model_type": "xgboost",
            "model_path": MODELS_DIR / "xgboost.pkl",
            "vol_penalty": 0.5,
        },
        "M3-E1b-RRF": {
            "label": "M3: 收益-夏普感知+RRF",
            "predictor_type": "fusion",
            "fusion_strategy": "rrf",
            "model_types": ["e1b_lightgbm", "xgboost"],
            "model_paths": [
                MODELS_DIR / "e1b_lightgbm.pkl",
                MODELS_DIR / "xgboost.pkl",
            ],
            "vol_penalty": 0.7,
            "k": 60,
        },
        "M4-E1a-AVG": {
            "label": "M4: 收益加权+分数平均",
            "predictor_type": "fusion",
            "fusion_strategy": "average",
            "model_types": ["e1a_lightgbm", "xgboost"],
            "model_paths": [
                MODELS_DIR / "e1a_lightgbm.pkl",
                MODELS_DIR / "xgboost.pkl",
            ],
            "vol_penalty": 0.5,
        },
    }


def create_predictor(config: dict[str, Any]) -> ModelPredictor | FusionPredictor:
    """
    根据方法配置创建预测器实例。

    对于融合方法，手动创建子预测器列表，再传入FusionPredictor。
    支持非标准模型类型（如e1b_lightgbm）通过model_path指定。

    参数:
        config: 方法配置字典

    返回:
        ModelPredictor或FusionPredictor实例

    异常:
        若模型文件不存在，抛出FileNotFoundError。
    """
    if config["predictor_type"] == "single":
        return ModelPredictor(
            model_type=config["model_type"],
            model_path=config["model_path"],
        )
    elif config["predictor_type"] == "fusion":
        sub_predictors: list[ModelPredictor] = []
        for mt, mp in zip(config["model_types"], config["model_paths"]):
            actual_type = "lightgbm" if "lightgbm" in mt else "xgboost"
            sub_predictors.append(ModelPredictor(model_type=actual_type, model_path=mp))

        fp = FusionPredictor(
            fusion_type=config["fusion_strategy"],
            model_types=["lightgbm", "xgboost"],
        )
        fp._predictors = sub_predictors
        fp._factor_cols = sub_predictors[0]._factor_cols
        return fp
    else:
        raise ValueError(f"未知的预测器类型: {config['predictor_type']}")


def run_backtest_for_method(
    name: str,
    config: dict[str, Any],
    weekly_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    对单个方法运行回测。

    参数:
        name: 方法名称
        config: 方法配置字典
        weekly_df: 周频数据

    返回:
        (回测结果DataFrame, 回测指标字典)
    """
    logger.info("运行回测: %s (%s)", name, config["label"])

    predictor = create_predictor(config)
    vol_penalty = config.get("vol_penalty", 0.0)

    engine = BacktestEngine(
        model_type=config.get("model_type", "lightgbm"),
        top_n=TOP_N,
        initial_capital=INITIAL_CAPITAL,
        vol_penalty=vol_penalty,
        custom_predictor=predictor,
    )

    result_df = engine.run_backtest(
        weekly_df,
        predictor=predictor,
        use_split="test",
    )

    metrics = compute_backtest_metrics(result_df, weekly_df, extended=True)

    logger.info(
        "  年化收益=%.2f%%, 夏普=%.4f, 最大回撤=%.2f%%",
        metrics["annualized_return"] * 100,
        metrics["sharpe_ratio"],
        metrics["max_drawdown"] * 100,
    )

    return result_df, metrics


def run_yearly_analysis(
    all_results: dict[str, pd.DataFrame],
) -> dict[str, dict[str, Any]]:
    """
    对每种方法进行分年度分析。

    参数:
        all_results: 方法名到回测结果DataFrame的映射

    返回:
        方法名到年度分析结果的映射
    """
    yearly_data: dict[str, dict[str, Any]] = {}

    for name, result_df in all_results.items():
        if result_df.empty or "date" not in result_df.columns:
            continue

        nav_series = result_df.set_index("date")["total_value"]
        if nav_series.empty:
            continue

        yearly_df = metrics_by_year(nav_series)
        if yearly_df.empty:
            continue

        yearly_data[name] = {}
        for year, row in yearly_df.iterrows():
            yearly_data[name][str(year)] = {
                "annualized_return": _safe_float(row.get("annualized_return")),
                "sharpe_ratio": _safe_float(row.get("sharpe_ratio")),
                "max_drawdown": _safe_float(row.get("max_drawdown")),
            }

        logger.info("分年度分析完成: %s, 覆盖%d年", name, len(yearly_data[name]))

    return yearly_data


def run_quantile_analysis(
    test_df: pd.DataFrame,
    all_results: dict[str, pd.DataFrame],
    configs: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """
    对每种方法进行分位数组合分析（多空组合收益）。

    参数:
        test_df: 测试集数据
        all_results: 方法名到回测结果DataFrame的映射
        configs: 方法配置字典

    返回:
        方法名到分位数分析结果的映射
    """
    quantile_data: dict[str, dict[str, Any]] = {}

    for name, config in configs.items():
        logger.info("分位数分析: %s", name)

        try:
            predictor = create_predictor(config)
            scores_df = predictor.predict_panel(test_df)

            if "future_return_1w" not in test_df.columns:
                logger.warning("测试集缺少future_return_1w列，跳过分位数分析: %s", name)
                continue

            merged = scores_df.merge(
                test_df[["date", "stock_code", "future_return_1w", "volatility_12w"]],
                on=["date", "stock_code"],
                how="left",
            )
            merged = merged.dropna(subset=["score", "future_return_1w"])

            vol_penalty = config.get("vol_penalty", 0.0)
            q_returns = quantile_portfolio_returns(
                merged, n_quantiles=N_QUANTILES,
                vol_penalty=vol_penalty, volatility_col="volatility_12w",
            )
            if not q_returns:
                logger.warning("分位数组合收益计算为空: %s", name)
                continue

            result: dict[str, Any] = {}
            for q_name, returns in q_returns.items():
                if len(returns) == 0:
                    continue
                ann_ret = float(np.mean(returns) * 52)
                sr = float(np.mean(returns) / (np.std(returns, ddof=1) + 1e-9) * np.sqrt(52))
                result[q_name] = {
                    "annualized_return": _safe_float(ann_ret),
                    "sharpe_ratio": _safe_float(sr),
                    "n_weeks": len(returns),
                }

            quantile_data[name] = result
            logger.info("  分位数分析完成: %s, %d个分组", name, len(result))

        except Exception as exc:
            logger.error("分位数分析失败: %s, 原因: %s", name, exc)

    return quantile_data


def run_significance_tests(
    all_results: dict[str, pd.DataFrame],
    baseline_name: str = "M1-B-LGBM",
) -> list[dict[str, Any]]:
    """
    对每种方法与基线进行配对显著性检验。

    参数:
        all_results: 方法名到回测结果DataFrame的映射
        baseline_name: 基线方法名

    返回:
        显著性检验结果列表
    """
    if baseline_name not in all_results:
        logger.warning("基线方法 %s 不在结果中，跳过显著性检验", baseline_name)
        return []

    baseline_returns = all_results[baseline_name]["weekly_return"].dropna().values
    tests: list[dict[str, Any]] = []

    for name, result_df in all_results.items():
        if name == baseline_name:
            continue
        if result_df.empty or "weekly_return" not in result_df.columns:
            continue

        method_returns = result_df["weekly_return"].dropna().values

        sharpe_ci = bootstrap_metric(
            method_returns,
            lambda r: np.mean(r) / (np.std(r, ddof=1) + 1e-9) * np.sqrt(52),
            n_bootstrap=N_BOOTSTRAP,
        )
        ann_ret_ci = bootstrap_metric(
            method_returns,
            lambda r: np.mean(r) * 52,
            n_bootstrap=N_BOOTSTRAP,
        )

        paired = paired_significance_test(method_returns, baseline_returns)

        tests.append({
            "method": name,
            "vs_baseline": baseline_name,
            "sharpe_ci": {k: _safe_float(v) for k, v in sharpe_ci.items()},
            "annualized_return_ci": {k: _safe_float(v) for k, v in ann_ret_ci.items()},
            "paired_test": {
                "t_stat": _safe_float(paired.get("t_stat")),
                "p_value": _safe_float(paired.get("p_value")),
                "significant": paired.get("significant", False),
                "method": paired.get("method", "unknown"),
            },
        })

        sig_mark = "***" if paired.get("p_value", 1) < 0.01 else (
            "**" if paired.get("p_value", 1) < 0.05 else (
                "*" if paired.get("p_value", 1) < 0.1 else ""
            )
        )
        logger.info(
            "显著性检验: %s vs %s, p=%.4f%s, 方法=%s",
            name, baseline_name,
            paired.get("p_value", 1),
            sig_mark,
            paired.get("method", "unknown"),
        )

    return tests


def generate_comprehensive_table(
    all_metrics: dict[str, dict[str, Any]],
    significance_tests: list[dict[str, Any]],
    configs: dict[str, dict[str, Any]],
) -> str:
    """
    生成综合对比表（Markdown格式）。

    参数:
        all_metrics: 方法名到回测指标的映射
        significance_tests: 显著性检验结果列表
        configs: 方法配置字典

    返回:
        Markdown格式的综合对比表
    """
    sig_map: dict[str, dict[str, Any]] = {}
    for t in significance_tests:
        sig_map[t["method"]] = t

    lines: list[str] = []
    lines.append("### 综合对比表")
    lines.append("")
    lines.append(
        "| 方法 | 损失函数 | 融合策略 | 年化收益(%) | 夏普比率 | 最大回撤(%) | "
        "换手率 | 胜率(%) | 夏普95%CI | vs M1 p值 |"
    )
    lines.append(
        "|------|---------|---------|------------|---------|------------|"
        "-------|--------|----------|----------|"
    )

    for name, config in configs.items():
        m = all_metrics.get(name, {})
        sig = sig_map.get(name, {})

        ann_ret = m.get("annualized_return", 0) * 100
        sr = m.get("sharpe_ratio", 0)
        md = m.get("max_drawdown", 0) * 100
        tr = m.get("turnover_rate", 0)
        wr = m.get("win_rate", 0) * 100

        loss_label = "标准NDCG"
        if "E1a" in name:
            loss_label = "收益加权"
        elif "E1b" in name:
            loss_label = "收益+夏普"
        elif "E1c" in name:
            loss_label = "收益+夏普+CVaR"

        fusion_label = "无"
        if "RRF" in name:
            fusion_label = "RRF"
        elif "AVG" in name:
            fusion_label = "分数平均"

        ci = sig.get("sharpe_ci", {})
        ci_str = "N/A"
        if ci and not np.isnan(ci.get("lower", float("nan"))):
            ci_str = f"[{ci['lower']:.2f}, {ci['upper']:.2f}]"

        p_val = sig.get("paired_test", {}).get("p_value", float("nan"))
        p_str = "—"
        if name != "M1-B-LGBM" and not np.isnan(p_val):
            sig_mark = "***" if p_val < 0.01 else (
                "**" if p_val < 0.05 else (
                    "*" if p_val < 0.1 else ""
                )
            )
            p_str = f"{p_val:.4f}{sig_mark}"

        lines.append(
            f"| {config['label']} | {loss_label} | {fusion_label} | "
            f"{ann_ret:.2f} | {sr:.4f} | {md:.2f} | {tr:.4f} | {wr:.2f} | "
            f"{ci_str} | {p_str} |"
        )

    lines.append("")
    lines.append("> * p<0.1, ** p<0.05, *** p<0.01（配对Wilcoxon/t检验）")
    lines.append("")
    return "\n".join(lines)


def generate_yearly_table(
    yearly_data: dict[str, dict[str, Any]],
    configs: dict[str, dict[str, Any]],
) -> str:
    """
    生成分年度分析表（Markdown格式）。

    参数:
        yearly_data: 方法名到年度分析结果的映射
        configs: 方法配置字典

    返回:
        Markdown格式的分年度分析表
    """
    all_years: set[str] = set()
    for name, yd in yearly_data.items():
        all_years.update(yd.keys())
    sorted_years = sorted(all_years)

    if not sorted_years:
        return "### 分年度分析\n\n数据不足，无法生成分年度分析表。\n"

    lines: list[str] = []
    lines.append("### 分年度分析")
    lines.append("")

    lines.append("#### 年化收益（%）")
    lines.append("")
    header = "| 年份 | 市场环境 |"
    sep = "|------|---------|"
    for name, config in configs.items():
        header += f" {config['label']} |"
        sep += "---------|"
    lines.append(header)
    lines.append(sep)

    env_map = {"2022": "熊市", "2023": "震荡", "2024": "先跌后涨"}
    for yr in sorted_years:
        env = env_map.get(yr, "—")
        row = f"| {yr} | {env} |"
        for name in configs:
            val = yearly_data.get(name, {}).get(yr, {}).get("annualized_return")
            row += f" {val * 100:.2f} |" if val is not None else " N/A |"
        lines.append(row)
    lines.append("")

    lines.append("#### 夏普比率")
    lines.append("")
    lines.append(header)
    lines.append(sep)
    for yr in sorted_years:
        env = env_map.get(yr, "—")
        row = f"| {yr} | {env} |"
        for name in configs:
            val = yearly_data.get(name, {}).get(yr, {}).get("sharpe_ratio")
            row += f" {val:.4f} |" if val is not None else " N/A |"
        lines.append(row)
    lines.append("")

    lines.append("#### 最大回撤（%）")
    lines.append("")
    lines.append(header)
    lines.append(sep)
    for yr in sorted_years:
        env = env_map.get(yr, "—")
        row = f"| {yr} | {env} |"
        for name in configs:
            val = yearly_data.get(name, {}).get(yr, {}).get("max_drawdown")
            row += f" {val * 100:.2f} |" if val is not None else " N/A |"
        lines.append(row)
    lines.append("")

    return "\n".join(lines)


def generate_quantile_table(
    quantile_data: dict[str, dict[str, Any]],
    configs: dict[str, dict[str, Any]],
) -> str:
    """
    生成多空组合分析表（Markdown格式）。

    参数:
        quantile_data: 方法名到分位数分析结果的映射
        configs: 方法配置字典

    返回:
        Markdown格式的多空组合分析表
    """
    lines: list[str] = []
    lines.append("### 多空组合收益分析")
    lines.append("")

    for name, config in configs.items():
        qd = quantile_data.get(name, {})
        if not qd:
            continue

        lines.append(f"#### {config['label']}")
        lines.append("")
        lines.append("| 分组 | 年化收益(%) | 夏普比率 |")
        lines.append("|------|------------|---------|")

        for q_name in ["Q1", "Q2", "Q3", "Q4", "Q5", "LS"]:
            q_info = qd.get(q_name, {})
            ann_ret = q_info.get("annualized_return")
            sr = q_info.get("sharpe_ratio")
            if ann_ret is not None and sr is not None:
                label = q_name
                if q_name == "LS":
                    label = "**多空(Q1-Q5)**"
                elif q_name == "Q1":
                    label = "Q1（最高分）"
                elif q_name == "Q5":
                    label = "Q5（最低分）"
                lines.append(f"| {label} | {ann_ret * 100:.2f} | {sr:.4f} |")

        lines.append("")

    return "\n".join(lines)


def generate_report(
    all_metrics: dict[str, dict[str, Any]],
    yearly_data: dict[str, dict[str, Any]],
    quantile_data: dict[str, dict[str, Any]],
    significance_tests: list[dict[str, Any]],
    configs: dict[str, dict[str, Any]],
) -> str:
    """
    生成实验4完整报告（Markdown格式）。

    参数:
        all_metrics: 各方法回测指标
        yearly_data: 分年度分析结果
        quantile_data: 多空组合分析结果
        significance_tests: 显著性检验结果
        configs: 方法配置字典

    返回:
        Markdown格式的完整报告文本
    """
    lines: list[str] = []
    lines.append("# 实验4：综合对比 + 分年度分析 + 多空组合 + 统计检验")
    lines.append("")
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    lines.append("## 1. 实验设计")
    lines.append("")
    lines.append("| 方法 | 损失函数 | 融合策略 | 说明 |")
    lines.append("|------|---------|---------|------|")
    lines.append("| M1 | 标准NDCG | 无（LightGBM单模型） | 基线 |")
    lines.append("| M2 | 标准NDCG | 无（XGBoost单模型） | 基线 |")
    lines.append("| M3 | 收益+夏普感知 | RRF融合 | 改进损失+融合 |")
    lines.append("| M4 | 收益加权 | 分数平均融合 | 改进损失+融合 |")
    lines.append("")

    lines.append("## 2. 综合对比")
    lines.append("")
    lines.append(generate_comprehensive_table(all_metrics, significance_tests, configs))

    lines.append("## 3. 分年度分析")
    lines.append("")
    lines.append(generate_yearly_table(yearly_data, configs))

    lines.append("## 4. 多空组合收益")
    lines.append("")
    lines.append(generate_quantile_table(quantile_data, configs))

    lines.append("## 5. 统计检验")
    lines.append("")
    lines.append("### Bootstrap置信区间（夏普比率）")
    lines.append("")
    lines.append("| 方法 | 点估计 | 95%置信区间 |")
    lines.append("|------|--------|-----------|")
    for t in significance_tests:
        ci = t.get("sharpe_ci", {})
        pt = ci.get("point", float("nan"))
        lo = ci.get("lower", float("nan"))
        up = ci.get("upper", float("nan"))
        if not np.isnan(pt):
            lines.append(f"| {t['method']} | {pt:.4f} | [{lo:.4f}, {up:.4f}] |")
        else:
            lines.append(f"| {t['method']} | N/A | N/A |")
    lines.append("")

    lines.append("### 配对显著性检验（vs M1基线）")
    lines.append("")
    lines.append("| 方法 | 检验方法 | t统计量 | p值 | 显著 |")
    lines.append("|------|---------|--------|-----|------|")
    for t in significance_tests:
        pt = t.get("paired_test", {})
        lines.append(
            f"| {t['method']} | {pt.get('method', 'N/A')} | "
            f"{_safe_float(pt.get('t_stat')):.4f} | "
            f"{_safe_float(pt.get('p_value')):.4f} | "
            f"{'是' if pt.get('significant') else '否'} |"
        )
    lines.append("")

    lines.append("## 6. 关键发现")
    lines.append("")

    best_method = max(all_metrics.items(), key=lambda x: x[1].get("sharpe_ratio", 0))
    best_name = best_method[0]
    best_sr = best_method[1].get("sharpe_ratio", 0)
    best_ar = best_method[1].get("annualized_return", 0) * 100
    lines.append(f"- **最优方法**: {configs[best_name]['label']}，夏普比率={best_sr:.4f}，年化收益={best_ar:.2f}%")

    baseline_sr = all_metrics.get("M1-B-LGBM", {}).get("sharpe_ratio", 0)
    if best_sr > baseline_sr:
        improvement = (best_sr - baseline_sr) / abs(baseline_sr + 1e-9) * 100
        lines.append(f"- **夏普比率提升**: 相比基线提升{improvement:.1f}%")

    sig_count = sum(1 for t in significance_tests if t.get("paired_test", {}).get("significant", False))
    lines.append(f"- **统计显著性**: {sig_count}/{len(significance_tests)}个方法与基线差异显著（p<0.05）")

    for name, qd in quantile_data.items():
        ls = qd.get("LS", {})
        if ls:
            ls_sr = ls.get("sharpe_ratio", 0)
            ls_ar = ls.get("annualized_return", 0)
            if ls_sr > 0:
                lines.append(
                    f"- **{configs[name]['label']}多空收益**: "
                    f"年化{ls_ar * 100:.2f}%, 夏普={ls_sr:.4f}，"
                    f"表明模型具有Alpha选股能力"
                )
    lines.append("")

    lines.append("## 7. 文件清单")
    lines.append("")
    lines.append("| 文件 | 说明 |")
    lines.append("|------|------|")
    lines.append("| experiment4_report.md | 本报告 |")
    lines.append("| experiment4_summary.json | 结构化摘要数据 |")
    lines.append("| yearly_analysis.json | 分年度分析数据 |")
    lines.append("| quantile_analysis.json | 多空组合分析数据 |")
    lines.append("| significance_tests.json | 统计检验数据 |")
    lines.append("")

    return "\n".join(lines)


def _safe_float(val: Any) -> float:
    """安全转换为float，NaN/None返回0.0。"""
    if val is None:
        return 0.0
    try:
        v = float(val)
        return v if np.isfinite(v) else 0.0
    except (TypeError, ValueError):
        return 0.0


def main() -> None:
    """实验4主函数。"""
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("实验4：综合对比 + 分年度分析 + 多空组合 + 统计检验")
    logger.info("=" * 60)

    configs = get_method_configurations()

    logger.info("步骤1：加载测试数据")
    test_df = load_weekly_test_data()

    logger.info("步骤2：运行各方法回测")
    all_metrics: dict[str, dict[str, Any]] = {}
    all_results: dict[str, pd.DataFrame] = {}

    for name, config in configs.items():
        try:
            result_df, metrics = run_backtest_for_method(name, config, test_df)
            all_metrics[name] = metrics
            all_results[name] = result_df

            result_path = EXPERIMENT_DIR / f"{name}_backtest_result.parquet"
            result_df.to_parquet(result_path, index=False)
            logger.info("回测结果已保存: %s", result_path)

            metrics_path = EXPERIMENT_DIR / f"{name}_backtest_metrics.json"
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2, default=str)

        except Exception as exc:
            logger.error("回测失败: %s, 原因: %s", name, exc)

    logger.info("步骤3：分年度分析")
    yearly_data = run_yearly_analysis(all_results)

    logger.info("步骤4：多空组合分析")
    quantile_data = run_quantile_analysis(test_df, all_results, configs)

    logger.info("步骤5：统计检验")
    significance_tests = run_significance_tests(all_results, baseline_name="M1-B-LGBM")

    logger.info("步骤6：生成报告")
    report = generate_report(all_metrics, yearly_data, quantile_data, significance_tests, configs)

    report_path = EXPERIMENT_DIR / "experiment4_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info("报告已保存: %s", report_path)

    summary: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "methods": {},
    }
    for name, metrics in all_metrics.items():
        summary["methods"][name] = {
            "annualized_return": _safe_float(metrics.get("annualized_return")),
            "sharpe_ratio": _safe_float(metrics.get("sharpe_ratio")),
            "max_drawdown": _safe_float(metrics.get("max_drawdown")),
            "turnover_rate": _safe_float(metrics.get("turnover_rate")),
            "win_rate": _safe_float(metrics.get("win_rate")),
        }

    summary_path = EXPERIMENT_DIR / "experiment4_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    yearly_path = EXPERIMENT_DIR / "yearly_analysis.json"
    with open(yearly_path, "w", encoding="utf-8") as f:
        json.dump(yearly_data, f, ensure_ascii=False, indent=2)

    quantile_path = EXPERIMENT_DIR / "quantile_analysis.json"
    with open(quantile_path, "w", encoding="utf-8") as f:
        json.dump(quantile_data, f, ensure_ascii=False, indent=2)

    sig_path = EXPERIMENT_DIR / "significance_tests.json"
    with open(sig_path, "w", encoding="utf-8") as f:
        json.dump(significance_tests, f, ensure_ascii=False, indent=2, default=str)

    logger.info("=" * 60)
    logger.info("实验4完成！")
    logger.info("方法数: %d", len(all_metrics))
    logger.info("年度分析: %d个方法", len(yearly_data))
    logger.info("多空分析: %d个方法", len(quantile_data))
    logger.info("显著性检验: %d组", len(significance_tests))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
