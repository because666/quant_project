"""
实验2：多模型排序融合对比

实验组：
- B-LGBM: LightGBM单模型（基线）
- B-XGB: XGBoost单模型（基线）
- E2a: 分数平均融合
- E2b: RRF融合（k=60）
- E2c: Stacking融合（Ridge元学习器）
- E2d: 加权RRF（验证集NDCG加权）

用法：
    cd backend
    python scripts/run_experiment2.py
"""
from __future__ import annotations

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

from src.fusion import FusionPredictor, stacking_fusion
from src.backtest import BacktestEngine, compute_backtest_metrics
from src.predictor import ModelPredictor
from src.data_loader import load_factor_columns, load_validation_data

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
EXPERIMENT_DIR = DATA_DIR / "experiment2"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("experiment2")

TOP_N: int = 20
INITIAL_CAPITAL: float = 1_000_000.0
VOL_PENALTY_GRID: list[float] = [0.0, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]


def get_raw_scores(predictor: ModelPredictor, X_df: pd.DataFrame) -> np.ndarray:
    """
    获取模型在给定因子矩阵上的原始预测分数。

    通过ModelPredictor内部接口直接调用Booster预测，
    避免predict方法中的排序和stock_code对齐开销。

    参数:
        predictor: 模型预测器实例
        X_df: 因子矩阵DataFrame（仅含因子列，不含stock_code）

    返回:
        与输入行对齐的原始预测分数数组，形状 (N,)
    """
    factor_df = X_df.copy()
    factor_df["stock_code"] = [f"dummy_{i}" for i in range(len(factor_df))]
    codes, X = predictor._align_factor_frame(factor_df)
    X_m = predictor._matrix(X)
    if predictor.model_type == "lightgbm":
        return np.asarray(predictor._booster.predict(X_m), dtype=np.float64)
    import xgboost as xgb
    dm = xgb.DMatrix(X_m, feature_names=predictor._factor_cols)
    return np.asarray(predictor._booster.predict(dm), dtype=np.float64)


def run_single_backtest(
    name: str,
    predictor: Any,
    weekly_df: pd.DataFrame,
    *,
    vol_penalty: float = 1.0,
    model_type: str = "lightgbm",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    使用指定预测器和参数运行单次回测。

    参数:
        name: 实验名称（用于日志）
        predictor: 预测器实例（ModelPredictor或FusionPredictor）
        weekly_df: 已加载的周频截面数据
        vol_penalty: 波动率惩罚系数
        model_type: BacktestEngine的model_type参数

    返回:
        (回测结果DataFrame, 回测指标字典)；结果为空时指标字典为空
    """
    t0 = time.time()
    engine = BacktestEngine(
        model_type,
        top_n=TOP_N,
        initial_capital=INITIAL_CAPITAL,
        vol_penalty=vol_penalty,
        custom_predictor=predictor,
    )
    result_df = engine.run_backtest(weekly_df, use_split="test")

    if result_df.empty:
        logger.warning("%s (vp=%.1f) 回测结果为空", name, vol_penalty)
        return result_df, {}

    metrics = compute_backtest_metrics(result_df, weekly_df, extended=True)
    elapsed = time.time() - t0

    logger.info(
        "%s (vp=%.1f) [%.1fs]: 年化=%.2f%%, 夏普=%.3f, 回撤=%.2f%%, 换手率=%.2f",
        name,
        vol_penalty,
        elapsed,
        metrics.get("annualized_return", 0) * 100,
        metrics.get("sharpe_ratio", 0),
        metrics.get("max_drawdown", 0) * 100,
        metrics.get("turnover_rate", 0),
    )

    return result_df, metrics


def search_optimal_vol_penalty(
    name: str,
    predictor: Any,
    weekly_df: pd.DataFrame,
    *,
    model_type: str = "lightgbm",
    grid: list[float] | None = None,
) -> tuple[float, dict[str, Any], pd.DataFrame]:
    """
    搜索最优vol_penalty参数（以夏普比率为优化目标）。

    参数:
        name: 实验名称
        predictor: 预测器实例
        weekly_df: 周频截面数据
        model_type: BacktestEngine的model_type
        grid: vol_penalty搜索网格，默认VOL_PENALTY_GRID

    返回:
        (最优vol_penalty, 最优指标字典, 最优回测结果DataFrame)
    """
    if grid is None:
        grid = VOL_PENALTY_GRID

    logger.info("搜索 %s 的最优vol_penalty，网格: %s", name, grid)

    best_vp: float = grid[0]
    best_metrics: dict[str, Any] = {}
    best_result: pd.DataFrame = pd.DataFrame()
    best_sharpe: float = float("-inf")

    for vp in grid:
        try:
            result_df, metrics = run_single_backtest(
                name,
                predictor,
                weekly_df,
                vol_penalty=vp,
                model_type=model_type,
            )
        except Exception as exc:
            logger.error("%s (vp=%.1f) 回测异常: %s", name, vp, exc)
            continue

        sharpe = metrics.get("sharpe_ratio", float("nan"))
        if isinstance(sharpe, (int, float)) and not np.isnan(sharpe) and sharpe > best_sharpe:
            best_sharpe = sharpe
            best_vp = vp
            best_metrics = metrics
            best_result = result_df

    logger.info("%s 最优vol_penalty=%.1f, 夏普=%.3f", name, best_vp, best_sharpe)
    return best_vp, best_metrics, best_result


def train_stacking_meta_learner() -> tuple[Any, dict[str, Any]]:
    """
    在验证集上训练Stacking元学习器（Ridge回归）。

    流程：
    1. 加载验证集因子矩阵与标签
    2. 分别用LightGBM和XGBoost模型对验证集预测
    3. 将两组预测分数作为元特征，训练Ridge回归

    返回:
        (训练好的Ridge模型实例, 元信息字典含系数/截距/样本数等)
    """
    logger.info("加载验证集数据...")
    val_X, val_y, val_groups = load_validation_data(fill_missing=True)

    logger.info("获取LightGBM验证集预测（%d样本）...", len(val_y))
    lgbm_pred = ModelPredictor("lightgbm")
    lgbm_val_scores = get_raw_scores(lgbm_pred, val_X)

    logger.info("获取XGBoost验证集预测...")
    xgb_pred = ModelPredictor("xgboost")
    xgb_val_scores = get_raw_scores(xgb_pred, val_X)

    logger.info("训练Ridge元学习器...")
    ridge_model, meta_info = stacking_fusion(
        [lgbm_val_scores, xgb_val_scores],
        val_y,
        val_groups,
    )

    logger.info(
        "Stacking元学习器训练完成: coefficients=%s, intercept=%.6f",
        meta_info.get("coefficients"),
        meta_info.get("intercept", 0),
    )

    return ridge_model, meta_info


def load_ndcg_weights() -> list[float]:
    """
    从模型指标JSON文件读取NDCG@10作为加权RRF的权重。

    返回:
        [LightGBM NDCG@10, XGBoost NDCG@10]
    """
    lgbm_metrics_path = MODELS_DIR / "lightgbm_metrics.json"
    xgb_metrics_path = MODELS_DIR / "xgboost_metrics.json"

    lgbm_ndcg: float = 0.0
    xgb_ndcg: float = 0.0

    if lgbm_metrics_path.exists():
        with open(lgbm_metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        lgbm_ndcg = data.get("val_ndcg", {}).get("ndcg@10", 0.0)
    else:
        logger.warning("LightGBM指标文件不存在: %s", lgbm_metrics_path)

    if xgb_metrics_path.exists():
        with open(xgb_metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        xgb_ndcg = data.get("val_ndcg", {}).get("ndcg@10", 0.0)
    else:
        logger.warning("XGBoost指标文件不存在: %s", xgb_metrics_path)

    logger.info("NDCG@10权重: LightGBM=%.4f, XGBoost=%.4f", lgbm_ndcg, xgb_ndcg)
    return [lgbm_ndcg, xgb_ndcg]


def generate_report(
    all_metrics: dict[str, dict[str, Any]],
    all_best_vp: dict[str, float],
    stacking_meta: dict[str, Any],
    ndcg_weights: list[float],
) -> str:
    """
    生成实验2对比报告（Markdown格式）。

    参数:
        all_metrics: 各实验组的回测指标 {name: metrics_dict}
        all_best_vp: 各实验组的最优vol_penalty {name: vp}
        stacking_meta: Stacking元学习器信息
        ndcg_weights: NDCG权重列表

    返回:
        Markdown格式的完整报告文本
    """
    lines: list[str] = []
    lines.append("# 实验2：多模型排序融合对比报告")
    lines.append("")
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    lines.append("## 1. 实验设计")
    lines.append("")
    lines.append("| 实验组 | 融合策略 | 核心思路 |")
    lines.append("|--------|---------|---------|")
    lines.append("| B-LGBM | 无融合（单模型） | LightGBM基线 |")
    lines.append("| B-XGB | 无融合（单模型） | XGBoost基线 |")
    lines.append("| E2a | 分数平均融合 | 多模型预测分数归一化后取平均 |")
    lines.append("| E2b | RRF融合（k=60） | Reciprocal Rank Fusion，按排名倒数加权 |")
    lines.append("| E2c | Stacking融合 | Ridge元学习器在验证集上学习最优融合权重 |")
    lines.append("| E2d | 加权RRF融合 | 根据验证集NDCG@10为各模型分配RRF权重 |")
    lines.append("")

    lines.append("## 2. 融合参数")
    lines.append("")
    lines.append("- RRF平滑常数 k=60")
    coeffs = stacking_meta.get("coefficients", [])
    if coeffs and len(coeffs) >= 2:
        lines.append(f"- Stacking元学习器系数: LightGBM={coeffs[0]:.6f}, XGBoost={coeffs[1]:.6f}")
    else:
        lines.append(f"- Stacking元学习器系数: {coeffs}")
    lines.append(f"- Stacking截距: {stacking_meta.get('intercept', 'N/A')}")
    lines.append(f"- NDCG@10权重: LightGBM={ndcg_weights[0]:.4f}, XGBoost={ndcg_weights[1]:.4f}")
    lines.append("")

    lines.append("## 3. 回测指标对比")
    lines.append("")
    lines.append("回测参数：top_n=20, use_split=test, vol_penalty为各策略最优值（按夏普比率择优）")
    lines.append("")

    header = "| 实验组 | 最优vol_penalty | 年化收益(%) | 夏普比率 | 最大回撤(%) | 换手率 | 周度胜率(%) |"
    sep = "|--------|---------------|------------|---------|------------|--------|------------|"
    lines.append(header)
    lines.append(sep)

    for name in ["B-LGBM", "B-XGB", "E2a", "E2b", "E2c", "E2d"]:
        m = all_metrics.get(name, {})
        vp = all_best_vp.get(name, 0)
        ann_ret = m.get("annualized_return", float("nan"))
        sharpe = m.get("sharpe_ratio", float("nan"))
        mdd = m.get("max_drawdown", float("nan"))
        turnover = m.get("turnover_rate", None)
        win_rate = m.get("win_rate", None)

        ann_ret_str = f"{ann_ret * 100:.2f}" if isinstance(ann_ret, (int, float)) and not np.isnan(ann_ret) else "N/A"
        sharpe_str = f"{sharpe:.3f}" if isinstance(sharpe, (int, float)) and not np.isnan(sharpe) else "N/A"
        mdd_str = f"{mdd * 100:.2f}" if isinstance(mdd, (int, float)) and not np.isnan(mdd) else "N/A"
        turnover_str = f"{turnover:.2f}" if turnover is not None else "N/A"
        win_rate_str = f"{win_rate * 100:.1f}" if win_rate is not None else "N/A"

        lines.append(f"| {name} | {vp:.1f} | {ann_ret_str} | {sharpe_str} | {mdd_str} | {turnover_str} | {win_rate_str} |")

    lines.append("")

    lines.append("## 4. 融合策略 vs 基线对比")
    lines.append("")

    baseline_lgbm = all_metrics.get("B-LGBM", {})
    baseline_xgb = all_metrics.get("B-XGB", {})
    baseline_ann_lgbm = baseline_lgbm.get("annualized_return", 0)
    baseline_sharpe_lgbm = baseline_lgbm.get("sharpe_ratio", 0)
    baseline_mdd_lgbm = baseline_lgbm.get("max_drawdown", 0)
    baseline_turnover_lgbm = baseline_lgbm.get("turnover_rate", 0)

    lines.append("### 4.1 vs B-LGBM基线")
    lines.append("")
    lines.append("| 对比 | 年化收益差(pp) | 夏普差 | 最大回撤差(pp) | 换手率差 | 是否优于基线 |")
    lines.append("|------|--------------|--------|--------------|---------|------------|")

    for name in ["E2a", "E2b", "E2c", "E2d"]:
        m = all_metrics.get(name, {})
        ann_ret = m.get("annualized_return", float("nan"))
        sharpe = m.get("sharpe_ratio", float("nan"))
        mdd = m.get("max_drawdown", float("nan"))
        turnover = m.get("turnover_rate", float("nan"))

        if np.isnan(ann_ret) or np.isnan(sharpe) or np.isnan(mdd):
            lines.append(f"| {name} vs B-LGBM | N/A | N/A | N/A | N/A | N/A |")
            continue

        ret_diff = (ann_ret - baseline_ann_lgbm) * 100
        sharpe_diff = sharpe - baseline_sharpe_lgbm
        mdd_diff = (mdd - baseline_mdd_lgbm) * 100
        turnover_diff = turnover - baseline_turnover_lgbm if not np.isnan(turnover) else float("nan")
        better = ann_ret > baseline_ann_lgbm and sharpe > baseline_sharpe_lgbm

        turnover_diff_str = f"{turnover_diff:+.2f}" if not np.isnan(turnover_diff) else "N/A"
        lines.append(
            f"| {name} vs B-LGBM | {ret_diff:+.2f} | {sharpe_diff:+.3f} "
            f"| {mdd_diff:+.2f} | {turnover_diff_str} | {'是' if better else '否'} |"
        )

    lines.append("")

    lines.append("### 4.2 RRF融合换手率分析")
    lines.append("")
    e2b_turnover = all_metrics.get("E2b", {}).get("turnover_rate", float("nan"))
    lgbm_turnover = baseline_lgbm.get("turnover_rate", float("nan"))

    if not np.isnan(e2b_turnover) and not np.isnan(lgbm_turnover):
        if e2b_turnover < lgbm_turnover:
            lines.append(f"E2b-RRF换手率（{e2b_turnover:.2f}）**低于** B-LGBM（{lgbm_turnover:.2f}），")
            lines.append(f"   差值: {e2b_turnover - lgbm_turnover:+.2f}，RRF融合有效降低了换手率。")
        else:
            lines.append(f"E2b-RRF换手率（{e2b_turnover:.2f}）**不低于** B-LGBM（{lgbm_turnover:.2f}），")
            lines.append(f"   差值: {e2b_turnover - lgbm_turnover:+.2f}，RRF融合未降低换手率。")
    else:
        lines.append("无法比较换手率（数据缺失）")

    lines.append("")

    lines.append("## 5. 关键论证逻辑")
    lines.append("")
    lines.append("### 5.1 实验假设")
    lines.append("")
    lines.append("单模型排序存在模型特定偏差，多模型融合可以通过互补性提升排序质量：")
    lines.append("- **E2a（分数平均）**：简单等权平均，假设各模型预测能力相当")
    lines.append("- **E2b（RRF）**：按排名倒数加权，对分数尺度不敏感，预期换手率更低")
    lines.append("- **E2c（Stacking）**：在验证集上学习最优线性组合权重")
    lines.append("- **E2d（加权RRF）**：根据NDCG@10为各模型分配不同权重")
    lines.append("")
    lines.append("### 5.2 预期效果")
    lines.append("")
    lines.append("1. RRF融合因基于排名而非分数，应降低组合换手率")
    lines.append("2. Stacking融合因学习了验证集上的最优权重，应提升排序质量")
    lines.append("3. 加权RRF应优于等权RRF（NDCG更高的模型获得更大权重）")
    lines.append("4. 分数平均融合作为最简单的融合方式，作为融合基线")
    lines.append("")
    lines.append("### 5.3 实验结论")
    lines.append("")

    e2b_metrics = all_metrics.get("E2b", {})
    e2b_better = False
    if e2b_metrics and baseline_lgbm:
        e2b_ann = e2b_metrics.get("annualized_return", float("nan"))
        e2b_sharpe = e2b_metrics.get("sharpe_ratio", float("nan"))
        if not np.isnan(e2b_ann) and not np.isnan(e2b_sharpe):
            e2b_better = e2b_ann > baseline_ann_lgbm and e2b_sharpe > baseline_sharpe_lgbm

    if e2b_better:
        lines.append("E2b-RRF融合**优于** B-LGBM基线，多模型排序融合有效。")
    else:
        lines.append("E2b-RRF融合**未优于** B-LGBM基线，可能原因：")
        lines.append("- 两模型排序高度相关，融合增益有限")
        lines.append("- RRF的排名聚合可能平滑掉了单模型的排序优势")
        lines.append("- 验证集与测试集分布差异导致Stacking权重过拟合")

    lines.append("")

    lines.append("## 6. 详细指标")
    lines.append("")
    for name in ["B-LGBM", "B-XGB", "E2a", "E2b", "E2c", "E2d"]:
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
    """
    主流程：训练Stacking元学习器 → 运行基线回测 → 运行融合回测 →
    搜索最优参数 → 生成报告。

    步骤：
    1. 在验证集上训练Ridge元学习器（供E2c使用）
    2. 读取NDCG@10权重（供E2d使用）
    3. 加载周频数据（只加载一次，所有回测共享）
    4. 运行基线回测（B-LGBM vp=1.0, B-XGB vp=0.5）
    5. 运行各融合策略回测（E2a/E2b/E2c/E2d，默认vp=1.0）
    6. 搜索各融合策略的最优vol_penalty
    7. 保存回测结果和指标
    8. 生成对比报告
    """
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("步骤1：训练Stacking元学习器")
    logger.info("=" * 60)
    try:
        ridge_model, stacking_meta = train_stacking_meta_learner()
        stacking_meta_path = EXPERIMENT_DIR / "stacking_meta.json"
        with open(stacking_meta_path, "w", encoding="utf-8") as f:
            json.dump(stacking_meta, f, ensure_ascii=False, indent=2, default=str)
        logger.info("Stacking元信息已保存: %s", stacking_meta_path)
    except Exception as exc:
        logger.error("Stacking元学习器训练失败: %s", exc)
        ridge_model = None
        stacking_meta = {"error": str(exc)}

    logger.info("=" * 60)
    logger.info("步骤2：加载NDCG权重")
    logger.info("=" * 60)
    ndcg_weights = load_ndcg_weights()

    logger.info("=" * 60)
    logger.info("步骤3：加载周频数据")
    logger.info("=" * 60)
    base_engine = BacktestEngine("lightgbm", top_n=TOP_N, initial_capital=INITIAL_CAPITAL)
    weekly_df = base_engine.load_weekly_data()
    logger.info("周频数据加载完成: %d行, %d列", len(weekly_df), len(weekly_df.columns))

    all_metrics: dict[str, dict[str, Any]] = {}
    all_best_vp: dict[str, float] = {}
    all_results: dict[str, pd.DataFrame] = {}

    logger.info("=" * 60)
    logger.info("步骤4：运行基线回测")
    logger.info("=" * 60)

    try:
        logger.info("--- B-LGBM (vp=1.0) ---")
        pred_lgbm = ModelPredictor("lightgbm")
        result_lgbm, m_lgbm = run_single_backtest(
            "B-LGBM", pred_lgbm, weekly_df, vol_penalty=1.0, model_type="lightgbm",
        )
        all_metrics["B-LGBM"] = m_lgbm
        all_best_vp["B-LGBM"] = 1.0
        all_results["B-LGBM"] = result_lgbm
    except Exception as exc:
        logger.error("B-LGBM回测失败: %s", exc)

    try:
        logger.info("--- B-XGB (vp=0.5) ---")
        pred_xgb = ModelPredictor("xgboost")
        result_xgb, m_xgb = run_single_backtest(
            "B-XGB", pred_xgb, weekly_df, vol_penalty=0.5, model_type="xgboost",
        )
        all_metrics["B-XGB"] = m_xgb
        all_best_vp["B-XGB"] = 0.5
        all_results["B-XGB"] = result_xgb
    except Exception as exc:
        logger.error("B-XGB回测失败: %s", exc)

    logger.info("=" * 60)
    logger.info("步骤5：运行融合策略回测（默认vp=1.0）")
    logger.info("=" * 60)

    try:
        logger.info("--- E2a: 分数平均融合 ---")
        fp_e2a = FusionPredictor("average", ["lightgbm", "xgboost"])
        result_e2a, m_e2a = run_single_backtest(
            "E2a", fp_e2a, weekly_df, vol_penalty=1.0, model_type="lightgbm",
        )
        all_metrics["E2a"] = m_e2a
        all_best_vp["E2a"] = 1.0
        all_results["E2a"] = result_e2a
    except Exception as exc:
        logger.error("E2a回测失败: %s", exc)

    try:
        logger.info("--- E2b: RRF融合 (k=60) ---")
        fp_e2b = FusionPredictor("rrf", ["lightgbm", "xgboost"], k=60)
        result_e2b, m_e2b = run_single_backtest(
            "E2b", fp_e2b, weekly_df, vol_penalty=1.0, model_type="lightgbm",
        )
        all_metrics["E2b"] = m_e2b
        all_best_vp["E2b"] = 1.0
        all_results["E2b"] = result_e2b
    except Exception as exc:
        logger.error("E2b回测失败: %s", exc)

    try:
        logger.info("--- E2c: Stacking融合 ---")
        if ridge_model is not None:
            fp_e2c = FusionPredictor("stacking", ["lightgbm", "xgboost"], ridge_model=ridge_model)
            result_e2c, m_e2c = run_single_backtest(
                "E2c", fp_e2c, weekly_df, vol_penalty=1.0, model_type="lightgbm",
            )
            all_metrics["E2c"] = m_e2c
            all_best_vp["E2c"] = 1.0
            all_results["E2c"] = result_e2c
        else:
            logger.error("E2c跳过：Ridge模型未训练成功")
    except Exception as exc:
        logger.error("E2c回测失败: %s", exc)

    try:
        logger.info("--- E2d: 加权RRF融合 ---")
        fp_e2d = FusionPredictor("weighted_rrf", ["lightgbm", "xgboost"], k=60, weights=ndcg_weights)
        result_e2d, m_e2d = run_single_backtest(
            "E2d", fp_e2d, weekly_df, vol_penalty=1.0, model_type="lightgbm",
        )
        all_metrics["E2d"] = m_e2d
        all_best_vp["E2d"] = 1.0
        all_results["E2d"] = result_e2d
    except Exception as exc:
        logger.error("E2d回测失败: %s", exc)

    logger.info("=" * 60)
    logger.info("步骤6：搜索各融合策略的最优vol_penalty")
    logger.info("=" * 60)

    fusion_predictors: dict[str, dict[str, Any]] = {}
    try:
        fusion_predictors["E2a"] = {
            "predictor": FusionPredictor("average", ["lightgbm", "xgboost"]),
            "model_type": "lightgbm",
        }
    except Exception as exc:
        logger.error("E2a预测器创建失败: %s", exc)

    try:
        fusion_predictors["E2b"] = {
            "predictor": FusionPredictor("rrf", ["lightgbm", "xgboost"], k=60),
            "model_type": "lightgbm",
        }
    except Exception as exc:
        logger.error("E2b预测器创建失败: %s", exc)

    try:
        if ridge_model is not None:
            fusion_predictors["E2c"] = {
                "predictor": FusionPredictor("stacking", ["lightgbm", "xgboost"], ridge_model=ridge_model),
                "model_type": "lightgbm",
            }
    except Exception as exc:
        logger.error("E2c预测器创建失败: %s", exc)

    try:
        fusion_predictors["E2d"] = {
            "predictor": FusionPredictor("weighted_rrf", ["lightgbm", "xgboost"], k=60, weights=ndcg_weights),
            "model_type": "lightgbm",
        }
    except Exception as exc:
        logger.error("E2d预测器创建失败: %s", exc)

    for name, cfg in fusion_predictors.items():
        logger.info("--- 搜索 %s ---", name)
        try:
            best_vp, best_metrics, best_result = search_optimal_vol_penalty(
                name,
                cfg["predictor"],
                weekly_df,
                model_type=cfg["model_type"],
            )
            all_best_vp[name] = best_vp
            all_metrics[name] = best_metrics
            all_results[name] = best_result
        except Exception as exc:
            logger.error("%s vol_penalty搜索失败: %s", name, exc)

    logger.info("=" * 60)
    logger.info("步骤7：保存结果")
    logger.info("=" * 60)

    for name, result_df in all_results.items():
        if not result_df.empty:
            safe_name = name.lower().replace("-", "_")
            result_path = EXPERIMENT_DIR / f"{safe_name}_backtest_result.parquet"
            result_df.to_parquet(result_path, index=False)
            logger.info("回测结果已保存: %s", result_path)

    for name, metrics in all_metrics.items():
        safe_name = name.lower().replace("-", "_")
        metrics_path = EXPERIMENT_DIR / f"{safe_name}_backtest_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2, default=str)
        logger.info("回测指标已保存: %s", metrics_path)

    logger.info("=" * 60)
    logger.info("步骤8：生成报告")
    logger.info("=" * 60)

    report = generate_report(all_metrics, all_best_vp, stacking_meta, ndcg_weights)
    report_path = EXPERIMENT_DIR / "experiment2_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info("实验报告已保存: %s", report_path)

    summary: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "models": {},
    }
    for name in ["B-LGBM", "B-XGB", "E2a", "E2b", "E2c", "E2d"]:
        m = all_metrics.get(name, {})
        summary["models"][name] = {
            "best_vol_penalty": all_best_vp.get(name),
            "annualized_return": m.get("annualized_return"),
            "sharpe_ratio": m.get("sharpe_ratio"),
            "max_drawdown": m.get("max_drawdown"),
            "turnover_rate": m.get("turnover_rate"),
            "win_rate": m.get("win_rate"),
        }

    summary_path = EXPERIMENT_DIR / "experiment2_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    logger.info("摘要已保存: %s", summary_path)

    logger.info("=" * 60)
    logger.info("实验2完成！核心结果摘要：")
    logger.info("=" * 60)
    for name in ["B-LGBM", "B-XGB", "E2a", "E2b", "E2c", "E2d"]:
        m = all_metrics.get(name, {})
        vp = all_best_vp.get(name, 0)
        ann_ret = m.get("annualized_return", float("nan"))
        sharpe = m.get("sharpe_ratio", float("nan"))
        mdd = m.get("max_drawdown", float("nan"))
        turnover = m.get("turnover_rate", float("nan"))
        logger.info(
            "%s (vp=%.1f): 年化=%.2f%%, 夏普=%.3f, 回撤=%.2f%%, 换手率=%.2f",
            name,
            vp,
            ann_ret * 100 if isinstance(ann_ret, (int, float)) and not np.isnan(ann_ret) else float("nan"),
            sharpe,
            mdd * 100 if isinstance(mdd, (int, float)) and not np.isnan(mdd) else float("nan"),
            turnover if isinstance(turnover, (int, float)) and not np.isnan(turnover) else float("nan"),
        )


if __name__ == "__main__":
    main()
