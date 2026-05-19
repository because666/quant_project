"""
统计检验 + 稳健性检验

根据实验方法论第五、六节，执行以下检验：
1. Bootstrap置信区间（年化收益、夏普比率、最大回撤）
2. 全配对显著性检验（所有方法两两比较）
3. Deflated Sharpe Ratio（校正多重试验偏差）
4. Top N参数敏感性分析（5, 10, 20, 30, 50）
5. 持有期敏感性分析（1周、2周、4周）
6. RRF平滑常数k敏感性分析（10, 30, 60, 100）

输出：
- data/statistical_tests/report.md
- data/statistical_tests/bootstrap_ci.json
- data/statistical_tests/paired_tests.json
- data/statistical_tests/deflated_sharpe.json
- data/statistical_tests/sensitivity_topn.json
- data/statistical_tests/sensitivity_holding.json
- data/statistical_tests/sensitivity_rrf_k.json
"""
from __future__ import annotations

import hashlib
import json
import logging
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest import BacktestEngine, compute_backtest_metrics
from src.metrics import (
    bootstrap_metric,
    max_drawdown,
    paired_significance_test,
    sharpe_ratio,
)
from src.predictor import ModelPredictor
from src.fusion import FusionPredictor

OUTPUT_DIR = PROJECT_ROOT / "data" / "statistical_tests"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS_DIR = PROJECT_ROOT / "models"
TOP_N = 20
INITIAL_CAPITAL = 1_000_000.0
N_BOOTSTRAP = 5000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUTPUT_DIR / "statistical_tests.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def _model_hash(path: Path) -> str:
    """计算模型文件的SHA256哈希前8位，用于校验加载了正确的模型。"""
    if not path.exists():
        return "MISSING"
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    return h[:8]


def _safe_float(val: Any) -> float:
    """安全转换为float，NaN/None返回0.0。"""
    if val is None:
        return 0.0
    try:
        v = float(val)
        return v if np.isfinite(v) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _sig_mark(p_val: float) -> str:
    """根据p值返回显著性标记。"""
    if np.isnan(p_val):
        return ""
    if p_val < 0.01:
        return "***"
    if p_val < 0.05:
        return "**"
    if p_val < 0.1:
        return "*"
    return ""


def deflated_sharpe_ratio(
    sharpe_annual: float,
    n_trials: int,
    skew: float,
    kurtosis: float,
    n_obs: int,
) -> float:
    """
    计算Deflated Sharpe Ratio，校正多重试验导致的夏普比率虚高。

    基于Bailey & Lopez de Prado (2014)的方法，
    考虑超参搜索次数、收益率偏度和峰度对夏普比率可信度的影响。

    参数:
        sharpe_annual: 观测到的年化夏普比率
        n_trials: 独立试验次数（如超参搜索的trial数）
        skew: 收益率分布的偏度
        kurtosis: 收益率分布的超额峰度
        n_obs: 观测期数（周数）

    返回:
        DSR值（0-1之间，>0.95表示夏普比率可信）
    """
    if n_trials < 1 or n_obs < 2:
        return 0.0

    sharpe_non_annual = sharpe_annual / np.sqrt(52)

    se_sq = (1 - skew * sharpe_non_annual + (kurtosis - 1) / 4 * sharpe_non_annual ** 2) / (n_obs - 1)
    if se_sq <= 0:
        se_sq = 1.0 / (n_obs - 1)
    se = np.sqrt(se_sq)

    expected_max_non_annual = (
        (1 - np.euler_gamma) * norm.ppf(1 - 1 / n_trials)
        + np.euler_gamma * norm.ppf(1 - 1 / (n_trials * np.e))
    ) / np.sqrt(52)

    dsr = float(norm.cdf((sharpe_non_annual - expected_max_non_annual) / se))
    return dsr


def get_method_configs() -> dict[str, dict[str, Any]]:
    """
    定义参与统计检验的方法配置。

    包含实验1-4中的所有关键方法。

    返回:
        方法名到配置字典的映射
    """
    return {
        "B-LGBM": {
            "label": "LightGBM基线",
            "predictor_type": "single",
            "model_type": "lightgbm",
            "model_path": MODELS_DIR / "lightgbm.pkl",
            "vol_penalty": 1.0,
        },
        "B-XGB": {
            "label": "XGBoost基线",
            "predictor_type": "single",
            "model_type": "xgboost",
            "model_path": MODELS_DIR / "xgboost.pkl",
            "vol_penalty": 0.5,
        },
        "E1a-LGBM": {
            "label": "收益加权NDCG",
            "predictor_type": "single",
            "model_type": "lightgbm",
            "model_path": MODELS_DIR / "e1a_lightgbm.pkl",
            "vol_penalty": 1.0,
        },
        "E1b-LGBM": {
            "label": "收益+夏普感知",
            "predictor_type": "single",
            "model_type": "lightgbm",
            "model_path": MODELS_DIR / "e1b_lightgbm.pkl",
            "vol_penalty": 0.5,
        },
        "E2a-AVG": {
            "label": "分数平均融合",
            "predictor_type": "fusion",
            "fusion_strategy": "average",
            "model_types": ["lightgbm", "xgboost"],
            "model_paths": [MODELS_DIR / "lightgbm.pkl", MODELS_DIR / "xgboost.pkl"],
            "vol_penalty": 0.5,
        },
        "E2b-RRF": {
            "label": "RRF融合",
            "predictor_type": "fusion",
            "fusion_strategy": "rrf",
            "model_types": ["lightgbm", "xgboost"],
            "model_paths": [MODELS_DIR / "lightgbm.pkl", MODELS_DIR / "xgboost.pkl"],
            "vol_penalty": 0.7,
            "k": 60,
        },
    }


def create_predictor(config: dict[str, Any]) -> ModelPredictor | FusionPredictor:
    """
    根据方法配置创建预测器实例。

    参数:
        config: 方法配置字典

    返回:
        ModelPredictor或FusionPredictor实例
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
            k=config.get("k", 60),
        )
        fp._predictors = sub_predictors
        fp._factor_cols = sub_predictors[0]._factor_cols
        return fp
    else:
        raise ValueError(f"未知的预测器类型: {config['predictor_type']}")


def run_all_backtests(
    configs: dict[str, dict[str, Any]],
    weekly_df: pd.DataFrame,
) -> dict[str, tuple[pd.DataFrame, dict[str, Any]]]:
    """
    对所有方法运行回测。

    参数:
        configs: 方法配置字典
        weekly_df: 周频数据

    返回:
        方法名到(回测结果DataFrame, 回测指标字典)的映射
    """
    results: dict[str, tuple[pd.DataFrame, dict[str, Any]]] = {}

    for name, config in configs.items():
        logger.info("回测: %s (%s)", name, config["label"])
        try:
            model_path = config.get("model_path")
            if model_path:
                logger.info("  模型文件: %s, SHA256前8位: %s", model_path, _model_hash(model_path))
            if config["predictor_type"] == "fusion":
                for i, mp in enumerate(config.get("model_paths", [])):
                    logger.info("  子模型[%d]: %s, SHA256前8位: %s", i, mp, _model_hash(mp))

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
            results[name] = (result_df, metrics)

            logger.info(
                "  年化=%.2f%%, 夏普=%.4f, 回撤=%.2f%%",
                metrics["annualized_return"] * 100,
                metrics["sharpe_ratio"],
                metrics["max_drawdown"] * 100,
            )
        except Exception as exc:
            logger.error("回测失败: %s, 原因: %s", name, exc)

    return results


def run_bootstrap_ci(
    all_results: dict[str, tuple[pd.DataFrame, dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """
    对所有方法计算Bootstrap置信区间。

    为年化收益、夏普比率、最大回撤三个核心指标构建95%置信区间。
    使用compute_backtest_metrics统一口径的夏普比率作为点估计。

    参数:
        all_results: 方法名到回测结果的映射

    返回:
        方法名到Bootstrap CI结果的映射
    """
    ci_data: dict[str, dict[str, Any]] = {}

    for name, (result_df, metrics) in all_results.items():
        if result_df.empty or "weekly_return" not in result_df.columns:
            continue

        returns = result_df["weekly_return"].dropna().values

        ann_ret_ci = bootstrap_metric(
            returns,
            lambda r: np.mean(r) * 52,
            n_bootstrap=N_BOOTSTRAP,
        )
        sharpe_ci = bootstrap_metric(
            returns,
            lambda r: np.mean(r) / (np.std(r, ddof=1) + 1e-9) * np.sqrt(52),
            n_bootstrap=N_BOOTSTRAP,
        )

        nav = result_df.set_index("date")["total_value"]
        if len(nav) > 2:
            nav_returns = nav.pct_change().dropna().values
            dd_ci = bootstrap_metric(
                nav_returns,
                lambda r: max_drawdown(pd.Series(np.cumprod(1 + r)))["max_drawdown"],
                n_bootstrap=N_BOOTSTRAP,
            )
        else:
            dd_ci = {"point": 0, "lower": 0, "upper": 0}

        ci_data[name] = {
            "annualized_return": {k: _safe_float(v) for k, v in ann_ret_ci.items()},
            "sharpe_ratio": {
                "point": _safe_float(metrics.get("sharpe_ratio", sharpe_ci.get("point", 0))),
                "lower": _safe_float(sharpe_ci.get("lower", 0)),
                "upper": _safe_float(sharpe_ci.get("upper", 0)),
            },
            "max_drawdown": {k: _safe_float(v) for k, v in dd_ci.items()},
        }

        logger.info(
            "Bootstrap CI: %s, 夏普=%.4f [%.4f, %.4f]（统一口径）",
            name,
            metrics.get("sharpe_ratio", 0),
            sharpe_ci.get("lower", 0),
            sharpe_ci.get("upper", 0),
        )

    return ci_data


def run_paired_tests(
    all_results: dict[str, tuple[pd.DataFrame, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """
    对所有方法进行全配对显著性检验。

    每对方法之间进行Wilcoxon/t检验，生成完整的p值矩阵。

    参数:
        all_results: 方法名到回测结果的映射

    返回:
        配对检验结果列表
    """
    names = list(all_results.keys())
    returns_map: dict[str, np.ndarray] = {}
    for name, (result_df, _) in all_results.items():
        if not result_df.empty and "weekly_return" in result_df.columns:
            returns_map[name] = result_df["weekly_return"].dropna().values

    tests: list[dict[str, Any]] = []
    for a_name, b_name in combinations(names, 2):
        if a_name not in returns_map or b_name not in returns_map:
            continue

        result = paired_significance_test(returns_map[a_name], returns_map[b_name])
        tests.append({
            "method_a": a_name,
            "method_b": b_name,
            "t_stat": _safe_float(result.get("t_stat")),
            "p_value": _safe_float(result.get("p_value")),
            "significant": result.get("significant", False),
            "test_method": result.get("method", "unknown"),
        })

        sm = _sig_mark(result.get("p_value", 1))
        logger.info(
            "配对检验: %s vs %s, p=%.4f%s (%s)",
            a_name, b_name,
            result.get("p_value", 1),
            sm,
            result.get("method", "unknown"),
        )

    return tests


def run_deflated_sharpe(
    all_results: dict[str, tuple[pd.DataFrame, dict[str, Any]]],
    n_trials: int = 20,
) -> dict[str, dict[str, Any]]:
    """
    对所有方法计算Deflated Sharpe Ratio。

    校正多重试验（超参搜索20次）导致的夏普比率虚高。

    参数:
        all_results: 方法名到回测结果的映射
        n_trials: 超参搜索的独立试验次数

    返回:
        方法名到DSR结果的映射
    """
    dsr_data: dict[str, dict[str, Any]] = {}

    for name, (result_df, metrics) in all_results.items():
        if result_df.empty or "weekly_return" not in result_df.columns:
            continue

        returns = result_df["weekly_return"].dropna().values
        sr = metrics.get("sharpe_ratio", 0)
        skew = float(stats.skew(returns))
        kurt = float(stats.kurtosis(returns))
        n_obs = len(returns)

        dsr = deflated_sharpe_ratio(sr, n_trials, skew, kurt, n_obs)

        dsr_data[name] = {
            "sharpe_ratio": _safe_float(sr),
            "n_trials": n_trials,
            "skewness": _safe_float(skew),
            "kurtosis": _safe_float(kurt),
            "n_obs": n_obs,
            "dsr": _safe_float(dsr),
            "credible": dsr > 0.95,
        }

        logger.info(
            "DSR: %s, 夏普=%.4f, DSR=%.4f, %s",
            name, sr, dsr,
            "可信" if dsr > 0.95 else "不可信",
        )

    return dsr_data


def run_topn_sensitivity(
    weekly_df: pd.DataFrame,
    top_n_list: list[int] | None = None,
) -> dict[str, dict[int, dict[str, Any]]]:
    """
    Top N参数敏感性分析。

    对每种方法测试不同选股数量（5, 10, 20, 30, 50）的回测表现。

    参数:
        weekly_df: 周频数据
        top_n_list: Top N测试列表

    返回:
        方法名到{TopN: 指标字典}的映射
    """
    if top_n_list is None:
        top_n_list = [5, 10, 20, 30, 50]

    configs = get_method_configs()
    sensitivity: dict[str, dict[int, dict[str, Any]]] = {}

    for name, config in configs.items():
        if config["predictor_type"] == "fusion":
            continue

        logger.info("Top N敏感性: %s", name)
        sensitivity[name] = {}

        predictor = create_predictor(config)
        vol_penalty = config.get("vol_penalty", 0.0)

        for top_n in top_n_list:
            try:
                engine = BacktestEngine(
                    model_type=config.get("model_type", "lightgbm"),
                    top_n=top_n,
                    initial_capital=INITIAL_CAPITAL,
                    vol_penalty=vol_penalty,
                    custom_predictor=predictor,
                )

                result_df = engine.run_backtest(weekly_df, predictor=predictor, use_split="test")
                metrics = compute_backtest_metrics(result_df, weekly_df, extended=True)

                sensitivity[name][top_n] = {
                    "annualized_return": _safe_float(metrics.get("annualized_return")),
                    "sharpe_ratio": _safe_float(metrics.get("sharpe_ratio")),
                    "max_drawdown": _safe_float(metrics.get("max_drawdown")),
                    "turnover_rate": _safe_float(metrics.get("turnover_rate")),
                }

                logger.info(
                    "  Top %d: 年化=%.2f%%, 夏普=%.4f",
                    top_n,
                    metrics.get("annualized_return", 0) * 100,
                    metrics.get("sharpe_ratio", 0),
                )
            except Exception as exc:
                logger.error("Top N=%d 失败: %s, 原因: %s", top_n, name, exc)

    return sensitivity


def run_holding_period_sensitivity(
    weekly_df: pd.DataFrame,
    holding_periods: list[int] | None = None,
) -> dict[str, dict[int, dict[str, Any]]]:
    """
    持有期敏感性分析。

    测试不同持有周期（1周、2周、4周）对回测结果的影响。
    使用BacktestEngine的rebalance_freq参数实现真正的多周持有逻辑。

    参数:
        weekly_df: 周频数据
        holding_periods: 持有期列表（单位：周）

    返回:
        方法名到{持有期: 指标字典}的映射
    """
    if holding_periods is None:
        holding_periods = [1, 2, 4]

    configs = get_method_configs()
    sensitivity: dict[str, dict[int, dict[str, Any]]] = {}

    for name, config in configs.items():
        if config["predictor_type"] == "fusion":
            continue

        logger.info("持有期敏感性: %s", name)
        sensitivity[name] = {}

        vol_penalty = config.get("vol_penalty", 0.0)
        predictor = create_predictor(config)

        for period in holding_periods:
            try:
                engine = BacktestEngine(
                    model_type=config.get("model_type", "lightgbm"),
                    top_n=TOP_N,
                    initial_capital=INITIAL_CAPITAL,
                    vol_penalty=vol_penalty,
                    rebalance_freq=period,
                    custom_predictor=predictor,
                )

                result_df = engine.run_backtest(weekly_df, predictor=predictor, use_split="test")

                if len(result_df) < 2:
                    continue

                metrics = compute_backtest_metrics(result_df, weekly_df, extended=True)

                sensitivity[name][period] = {
                    "annualized_return": _safe_float(metrics.get("annualized_return")),
                    "sharpe_ratio": _safe_float(metrics.get("sharpe_ratio")),
                    "max_drawdown": _safe_float(metrics.get("max_drawdown")),
                    "turnover_rate": _safe_float(metrics.get("turnover_rate")),
                }

                logger.info(
                    "  持有%d周: 年化=%.2f%%, 夏普=%.4f, 换手率=%.4f",
                    period,
                    metrics.get("annualized_return", 0) * 100,
                    metrics.get("sharpe_ratio", 0),
                    metrics.get("turnover_rate", 0),
                )
            except Exception as exc:
                logger.error("持有期=%d周 失败: %s, 原因: %s", period, name, exc)

    return sensitivity


def run_rrf_k_sensitivity(
    weekly_df: pd.DataFrame,
    k_list: list[int] | None = None,
) -> dict[int, dict[str, Any]]:
    """
    RRF平滑常数k敏感性分析。

    测试不同k值（10, 30, 60, 100）对RRF融合回测结果的影响。

    参数:
        weekly_df: 周频数据
        k_list: k值测试列表

    返回:
        k值到指标字典的映射
    """
    if k_list is None:
        k_list = [10, 30, 60, 100]

    sensitivity: dict[int, dict[str, Any]] = {}

    for k in k_list:
        logger.info("RRF k敏感性: k=%d", k)
        try:
            lgb_pred = ModelPredictor(model_type="lightgbm", model_path=MODELS_DIR / "lightgbm.pkl")
            xgb_pred = ModelPredictor(model_type="xgboost", model_path=MODELS_DIR / "xgboost.pkl")

            fp = FusionPredictor(fusion_type="rrf", model_types=["lightgbm", "xgboost"], k=k)
            fp._predictors = [lgb_pred, xgb_pred]
            fp._factor_cols = lgb_pred._factor_cols

            engine = BacktestEngine(
                model_type="lightgbm",
                top_n=TOP_N,
                initial_capital=INITIAL_CAPITAL,
                vol_penalty=0.7,
                custom_predictor=fp,
            )

            result_df = engine.run_backtest(weekly_df, predictor=fp, use_split="test")
            metrics = compute_backtest_metrics(result_df, weekly_df, extended=True)

            sensitivity[k] = {
                "annualized_return": _safe_float(metrics.get("annualized_return")),
                "sharpe_ratio": _safe_float(metrics.get("sharpe_ratio")),
                "max_drawdown": _safe_float(metrics.get("max_drawdown")),
                "turnover_rate": _safe_float(metrics.get("turnover_rate")),
            }

            logger.info(
                "  k=%d: 年化=%.2f%%, 夏普=%.4f",
                k,
                metrics.get("annualized_return", 0) * 100,
                metrics.get("sharpe_ratio", 0),
            )
        except Exception as exc:
            logger.error("RRF k=%d 失败, 原因: %s", k, exc)

    return sensitivity


def generate_report(
    ci_data: dict[str, dict[str, Any]],
    paired_tests: list[dict[str, Any]],
    dsr_data: dict[str, dict[str, Any]],
    topn_sens: dict[str, dict[int, dict[str, Any]]],
    holding_sens: dict[str, dict[int, dict[str, Any]]],
    rrf_k_sens: dict[int, dict[str, Any]],
    configs: dict[str, dict[str, Any]],
    n_trials: int = 20,
) -> str:
    """
    生成统计检验+稳健性检验完整报告（Markdown格式）。

    参数:
        ci_data: Bootstrap CI数据
        paired_tests: 配对检验数据
        dsr_data: DSR数据
        topn_sens: Top N敏感性数据
        holding_sens: 持有期敏感性数据
        rrf_k_sens: RRF k敏感性数据
        configs: 方法配置字典

    返回:
        Markdown格式的完整报告文本
    """
    lines: list[str] = []
    lines.append("# 统计检验 + 稳健性检验报告")
    lines.append("")
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # ── 统计检验 ──
    lines.append("## 一、统计检验")
    lines.append("")

    # 1. Bootstrap CI
    lines.append("### 1.1 Bootstrap置信区间（95%）")
    lines.append("")
    lines.append("| 方法 | 年化收益(%) | CI | 夏普比率 | CI | 最大回撤(%) | CI |")
    lines.append("|------|------------|-----|---------|-----|------------|-----|")
    for name, ci in ci_data.items():
        ar = ci.get("annualized_return", {})
        sr = ci.get("sharpe_ratio", {})
        dd = ci.get("max_drawdown", {})
        label = configs.get(name, {}).get("label", name)
        lines.append(
            f"| {label} | {ar.get('point', 0) * 100:.2f} | "
            f"[{ar.get('lower', 0) * 100:.2f}, {ar.get('upper', 0) * 100:.2f}] | "
            f"{sr.get('point', 0):.4f} | "
            f"[{sr.get('lower', 0):.4f}, {sr.get('upper', 0):.4f}] | "
            f"{dd.get('point', 0) * 100:.2f} | "
            f"[{dd.get('lower', 0) * 100:.2f}, {dd.get('upper', 0) * 100:.2f}] |"
        )
    lines.append("")

    # 2. 配对检验
    lines.append("### 1.2 配对显著性检验（全配对）")
    lines.append("")
    lines.append("| 方法A | 方法B | 检验方法 | p值 | 显著 |")
    lines.append("|-------|-------|---------|-----|------|")
    for t in paired_tests:
        a_label = configs.get(t["method_a"], {}).get("label", t["method_a"])
        b_label = configs.get(t["method_b"], {}).get("label", t["method_b"])
        sm = _sig_mark(t["p_value"])
        lines.append(
            f"| {a_label} | {b_label} | {t['test_method']} | "
            f"{t['p_value']:.4f}{sm} | {'是' if t['significant'] else '否'} |"
        )
    lines.append("")
    lines.append("> * p<0.1, ** p<0.05, *** p<0.01")
    lines.append("")

    # 3. DSR
    lines.append("### 1.3 Deflated Sharpe Ratio")
    lines.append("")
    lines.append("| 方法 | 夏普比率 | DSR | 可信(>0.95) | 偏度 | 峰度 | n_obs |")
    lines.append("|------|---------|-----|------------|------|------|-------|")
    for name, d in dsr_data.items():
        label = configs.get(name, {}).get("label", name)
        lines.append(
            f"| {label} | {d['sharpe_ratio']:.4f} | {d['dsr']:.4f} | "
            f"{'✅' if d['credible'] else '❌'} | "
            f"{d['skewness']:.4f} | {d['kurtosis']:.4f} | {d['n_obs']} |"
        )
    lines.append("")
    lines.append(
        f"> DSR校正了{n_trials}次超参搜索的多重试验偏差。"
        f"DSR>0.95表示夏普比率在统计上可信。"
    )
    lines.append("")

    # ── 稳健性检验 ──
    lines.append("## 二、稳健性检验")
    lines.append("")

    # 4. Top N敏感性
    lines.append("### 2.1 Top N参数敏感性")
    lines.append("")
    for name, topn_data in topn_sens.items():
        label = configs.get(name, {}).get("label", name)
        lines.append(f"#### {label}")
        lines.append("")
        lines.append("| Top N | 年化收益(%) | 夏普比率 | 最大回撤(%) |")
        lines.append("|-------|------------|---------|------------|")
        for top_n in sorted(topn_data.keys()):
            d = topn_data[top_n]
            lines.append(
                f"| {top_n} | {d['annualized_return'] * 100:.2f} | "
                f"{d['sharpe_ratio']:.4f} | {d['max_drawdown'] * 100:.2f} |"
            )
        lines.append("")

    # 5. 持有期敏感性
    lines.append("### 2.2 持有期敏感性")
    lines.append("")
    for name, holding_data in holding_sens.items():
        label = configs.get(name, {}).get("label", name)
        lines.append(f"#### {label}")
        lines.append("")
        lines.append("| 持有期(周) | 年化收益(%) | 夏普比率 | 最大回撤(%) |")
        lines.append("|-----------|------------|---------|------------|")
        for period in sorted(holding_data.keys()):
            d = holding_data[period]
            lines.append(
                f"| {period} | {d['annualized_return'] * 100:.2f} | "
                f"{d['sharpe_ratio']:.4f} | {d['max_drawdown'] * 100:.2f} |"
            )
        lines.append("")

    # 6. RRF k敏感性
    lines.append("### 2.3 RRF平滑常数k敏感性")
    lines.append("")
    lines.append("| k值 | 年化收益(%) | 夏普比率 | 最大回撤(%) |")
    lines.append("|-----|------------|---------|------------|")
    for k in sorted(rrf_k_sens.keys()):
        d = rrf_k_sens[k]
        lines.append(
            f"| {k} | {d['annualized_return'] * 100:.2f} | "
            f"{d['sharpe_ratio']:.4f} | {d['max_drawdown'] * 100:.2f} |"
        )
    lines.append("")

    # ── 关键结论 ──
    lines.append("## 三、关键结论")
    lines.append("")

    credible_count = sum(1 for d in dsr_data.values() if d["credible"])
    lines.append(f"- **DSR可信性**: {credible_count}/{len(dsr_data)}个方法的夏普比率通过DSR检验（>0.95）")

    sig_count = sum(1 for t in paired_tests if t["significant"])
    lines.append(f"- **配对检验**: {sig_count}/{len(paired_tests)}组方法对差异显著（p<0.05）")

    if topn_sens:
        first_method = list(topn_sens.keys())[0]
        sharpe_values = [topn_sens[first_method][tn]["sharpe_ratio"] for tn in sorted(topn_sens[first_method])]
        if sharpe_values:
            sharpe_range = max(sharpe_values) - min(sharpe_values)
            lines.append(
                f"- **Top N敏感性**: {configs.get(first_method, {}).get('label', first_method)}的夏普比率"
                f"在Top 5~50范围内波动{sharpe_range:.4f}，"
                f"{'参数敏感性较低' if sharpe_range < 0.5 else '参数敏感性较高'}"
            )

    if rrf_k_sens:
        rrf_sharpes = [rrf_k_sens[k]["sharpe_ratio"] for k in sorted(rrf_k_sens.keys())]
        rrf_range = max(rrf_sharpes) - min(rrf_sharpes)
        lines.append(
            f"- **RRF k敏感性**: 夏普比率在k=10~100范围内波动{rrf_range:.4f}，"
            f"{'对k值不敏感' if rrf_range < 0.3 else '对k值较敏感'}"
        )

    lines.append("")

    lines.append("## 四、文件清单")
    lines.append("")
    lines.append("| 文件 | 说明 |")
    lines.append("|------|------|")
    lines.append("| report.md | 本报告 |")
    lines.append("| bootstrap_ci.json | Bootstrap置信区间数据 |")
    lines.append("| paired_tests.json | 配对显著性检验数据 |")
    lines.append("| deflated_sharpe.json | Deflated Sharpe Ratio数据 |")
    lines.append("| sensitivity_topn.json | Top N敏感性数据 |")
    lines.append("| sensitivity_holding.json | 持有期敏感性数据 |")
    lines.append("| sensitivity_rrf_k.json | RRF k敏感性数据 |")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    """统计检验+稳健性检验主函数。"""
    logger.info("=" * 60)
    logger.info("统计检验 + 稳健性检验")
    logger.info("=" * 60)

    configs = get_method_configs()

    logger.info("步骤1：加载测试数据")
    engine = BacktestEngine(model_type="lightgbm", top_n=TOP_N, initial_capital=INITIAL_CAPITAL)
    weekly_df = engine.load_weekly_data(concat_splits=True)
    logger.info("数据加载完成: %d行", len(weekly_df))

    logger.info("步骤2：运行所有方法回测")
    all_results = run_all_backtests(configs, weekly_df)
    logger.info("回测完成: %d个方法", len(all_results))

    logger.info("步骤3：Bootstrap置信区间")
    ci_data = run_bootstrap_ci(all_results)

    logger.info("步骤4：全配对显著性检验")
    paired_tests = run_paired_tests(all_results)

    logger.info("步骤5：Deflated Sharpe Ratio")
    n_trials = 20
    dsr_data = run_deflated_sharpe(all_results, n_trials=n_trials)

    logger.info("步骤6：Top N参数敏感性")
    topn_sens = run_topn_sensitivity(weekly_df)

    logger.info("步骤7：持有期敏感性")
    holding_sens = run_holding_period_sensitivity(weekly_df)

    logger.info("步骤8：RRF k敏感性")
    rrf_k_sens = run_rrf_k_sensitivity(weekly_df)

    logger.info("步骤9：生成报告")
    report = generate_report(
        ci_data, paired_tests, dsr_data,
        topn_sens, holding_sens, rrf_k_sens,
        configs, n_trials=n_trials,
    )

    report_path = OUTPUT_DIR / "report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    for filename, data in [
        ("bootstrap_ci.json", ci_data),
        ("paired_tests.json", paired_tests),
        ("deflated_sharpe.json", dsr_data),
        ("sensitivity_topn.json", topn_sens),
        ("sensitivity_holding.json", holding_sens),
        ("sensitivity_rrf_k.json", rrf_k_sens),
    ]:
        path = OUTPUT_DIR / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    logger.info("=" * 60)
    logger.info("统计检验+稳健性检验完成！")
    logger.info("方法数: %d", len(all_results))
    logger.info("配对检验: %d组", len(paired_tests))
    logger.info("DSR: %d个方法", len(dsr_data))
    logger.info("Top N敏感性: %d个方法", len(topn_sens))
    logger.info("持有期敏感性: %d个方法", len(holding_sens))
    logger.info("RRF k敏感性: %d个k值", len(rrf_k_sens))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
