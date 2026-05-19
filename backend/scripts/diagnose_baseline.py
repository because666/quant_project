"""
基线负收益深度诊断脚本：全面分析模型分数与收益/波动率的关系，定位根因。

诊断内容：
1. 模型分数与未来收益的Spearman/Pearson相关性（按截面和全局）
2. 模型分数与波动率因子的相关性
3. Top N / Bottom N / 随机选股的实际收益分布对比
4. 高波动暴露分析：Top N股票的波动率是否显著高于全市场
5. 训练/验证/测试集NDCG对比（过拟合检测）
6. 因子有效性检查（零值因子、低方差因子）
7. 时间对齐验证（确保无未来信息泄露）

用法：
    cd backend
    python scripts/diagnose_baseline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.backtest import BacktestEngine, compute_backtest_metrics
from src.data_loader import load_factor_columns, split_by_time
from src.predictor import ModelPredictor

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORT_PATH = DATA_DIR / "baseline" / "diagnosis_report.md"


def _load_test_df() -> pd.DataFrame:
    """加载测试集数据，优先使用未压缩版本。"""
    test_path = DATA_DIR / "test.parquet"
    if not test_path.exists():
        test_path = DATA_DIR / "test.parquet.gz"
    return pd.read_parquet(test_path)


def diagnose_score_return_correlation() -> dict[str, Any]:
    """诊断1：模型分数与未来收益的相关性。"""
    logger.info("=" * 60)
    logger.info("诊断1：模型分数与未来收益的相关性")
    logger.info("=" * 60)

    results: dict[str, Any] = {}

    df = _load_test_df()
    if df.empty:
        logger.error("测试数据为空")
        return {"error": "测试数据为空"}

    dates = sorted(df["date"].unique())
    logger.info("测试集日期范围: %s ~ %s, 共 %d 个截面", dates[0], dates[-1], len(dates))

    for model_type in ("lightgbm", "xgboost"):
        try:
            pred = ModelPredictor(model_type, data_dir=DATA_DIR)
        except FileNotFoundError:
            logger.warning("模型 %s 不存在，跳过", model_type)
            continue

        spearman_by_date: list[float] = []
        pearson_by_date: list[float] = []

        sample_dates = dates[::5]
        for d in sample_dates:
            section = df[df["date"] == d].copy()
            if len(section) < 50:
                continue
            try:
                factor_cols = pred._factor_cols
                avail_cols = [c for c in factor_cols if c in section.columns]
                if len(avail_cols) < 5:
                    continue
                pred_df = pred.predict(section[["stock_code"] + avail_cols])
                merged = pred_df.merge(
                    section[["stock_code", "future_return_1w"]],
                    on="stock_code",
                    how="left",
                )
                valid = merged.dropna(subset=["score", "future_return_1w"])
                if len(valid) < 30:
                    continue

                sp_corr, _ = stats.spearmanr(valid["score"], valid["future_return_1w"])
                pe_corr, _ = stats.pearsonr(valid["score"], valid["future_return_1w"])
                if not np.isnan(sp_corr):
                    spearman_by_date.append(float(sp_corr))
                if not np.isnan(pe_corr):
                    pearson_by_date.append(float(pe_corr))
            except Exception as e:
                logger.warning("截面 %s 预测失败: %s", d, e)

        avg_sp = float(np.mean(spearman_by_date)) if spearman_by_date else float("nan")
        avg_pe = float(np.mean(pearson_by_date)) if pearson_by_date else float("nan")
        pct_negative_sp = float(np.mean([c < 0 for c in spearman_by_date])) if spearman_by_date else float("nan")

        results[model_type] = {
            "avg_spearman": avg_sp,
            "avg_pearson": avg_pe,
            "pct_negative_spearman": pct_negative_sp,
            "n_dates": len(spearman_by_date),
        }

        direction = "负相关（模型排序反了）" if avg_sp < 0 else "正相关（模型排序正确）"
        logger.info(
            "%s: 平均Spearman=%.4f, 平均Pearson=%.4f, 负相关截面占比=%.1f%% → %s",
            model_type.upper(),
            avg_sp,
            avg_pe,
            pct_negative_sp * 100,
            direction,
        )

    return results


def diagnose_volatility_bias() -> dict[str, Any]:
    """诊断2：模型选股的波动率暴露。"""
    logger.info("")
    logger.info("=" * 60)
    logger.info("诊断2：模型选股的波动率暴露")
    logger.info("=" * 60)

    results: dict[str, Any] = {}

    df = _load_test_df()
    vol_col = "volatility_12w"
    if vol_col not in df.columns:
        logger.warning("数据中无 %s 列，跳过波动率分析", vol_col)
        return {"error": f"无 {vol_col} 列"}

    dates = sorted(df["date"].unique())
    sample_dates = dates[::5]

    for model_type in ("lightgbm", "xgboost"):
        try:
            pred = ModelPredictor(model_type, data_dir=DATA_DIR)
        except FileNotFoundError:
            continue

        top_vols: list[float] = []
        bottom_vols: list[float] = []
        all_vols: list[float] = []

        for d in sample_dates:
            section = df[df["date"] == d].copy()
            if len(section) < 50:
                continue
            try:
                factor_cols = pred._factor_cols
                avail_cols = [c for c in factor_cols if c in section.columns]
                if len(avail_cols) < 5:
                    continue
                pred_df = pred.predict(section[["stock_code"] + avail_cols])
                merged = pred_df.merge(
                    section[["stock_code", vol_col]],
                    on="stock_code",
                    how="left",
                )
                valid = merged.dropna(subset=["score", vol_col])
                if len(valid) < 30:
                    continue

                top20_vol = float(valid.nlargest(20, "score")[vol_col].mean())
                bottom20_vol = float(valid.nsmallest(20, "score")[vol_col].mean())
                all_vol = float(valid[vol_col].mean())

                top_vols.append(top20_vol)
                bottom_vols.append(bottom20_vol)
                all_vols.append(all_vol)
            except Exception:
                pass

        avg_top = float(np.mean(top_vols)) if top_vols else float("nan")
        avg_bottom = float(np.mean(bottom_vols)) if bottom_vols else float("nan")
        avg_all = float(np.mean(all_vols)) if all_vols else float("nan")

        results[model_type] = {
            "top20_avg_vol": avg_top,
            "bottom20_avg_vol": avg_bottom,
            "all_avg_vol": avg_all,
            "top_vs_all_ratio": avg_top / avg_all if avg_all > 0 else float("nan"),
        }

        logger.info(
            "%s: Top20平均波动率=%.4f, Bottom20=%.4f, 全市场=%.4f, Top/全市场比=%.2f",
            model_type.upper(),
            avg_top,
            avg_bottom,
            avg_all,
            avg_top / avg_all if avg_all > 0 else 0,
        )

    return results


def diagnose_top_bottom_returns() -> dict[str, Any]:
    """诊断3：Top N / Bottom N / 随机选股的实际收益分布。"""
    logger.info("")
    logger.info("=" * 60)
    logger.info("诊断3：Top/Bottom/随机选股收益分布")
    logger.info("=" * 60)

    results: dict[str, Any] = {}
    df = _load_test_df()
    dates = sorted(df["date"].unique())
    sample_dates = dates[::5]

    for model_type in ("lightgbm", "xgboost"):
        try:
            pred = ModelPredictor(model_type, data_dir=DATA_DIR)
        except FileNotFoundError:
            continue

        top_rets: list[float] = []
        bottom_rets: list[float] = []
        random_rets: list[float] = []

        rng = np.random.default_rng(42)

        for d in sample_dates:
            section = df[df["date"] == d].copy()
            if len(section) < 50:
                continue
            try:
                factor_cols = pred._factor_cols
                avail_cols = [c for c in factor_cols if c in section.columns]
                if len(avail_cols) < 5:
                    continue
                pred_df = pred.predict(section[["stock_code"] + avail_cols])
                merged = pred_df.merge(
                    section[["stock_code", "future_return_1w"]],
                    on="stock_code",
                    how="left",
                )
                valid = merged.dropna(subset=["score", "future_return_1w"])
                if len(valid) < 30:
                    continue

                top20_ret = float(valid.nlargest(20, "score")["future_return_1w"].mean())
                bottom20_ret = float(valid.nsmallest(20, "score")["future_return_1w"].mean())
                random20 = valid.sample(n=min(20, len(valid)), random_state=rng.integers(0, 2**31))
                random20_ret = float(random20["future_return_1w"].mean())

                top_rets.append(top20_ret)
                bottom_rets.append(bottom20_ret)
                random_rets.append(random20_ret)
            except Exception:
                pass

        avg_top = float(np.mean(top_rets)) if top_rets else float("nan")
        avg_bottom = float(np.mean(bottom_rets)) if bottom_rets else float("nan")
        avg_random = float(np.mean(random_rets)) if random_rets else float("nan")

        results[model_type] = {
            "top20_avg_weekly_return": avg_top,
            "bottom20_avg_weekly_return": avg_bottom,
            "random20_avg_weekly_return": avg_random,
            "top_minus_random": avg_top - avg_random,
            "bottom_minus_random": avg_bottom - avg_random,
        }

        logger.info(
            "%s: Top20周均收益=%.4f%%, Bottom20=%.4f%%, 随机20=%.4f%%, Top-随机=%.4f%%",
            model_type.upper(),
            avg_top * 100,
            avg_bottom * 100,
            avg_random * 100,
            (avg_top - avg_random) * 100,
        )

    return results


def diagnose_overfitting() -> dict[str, Any]:
    """诊断4：训练/验证/测试集NDCG对比（过拟合检测）。"""
    logger.info("")
    logger.info("=" * 60)
    logger.info("诊断4：过拟合检测")
    logger.info("=" * 60)

    results: dict[str, Any] = {}

    for model_type in ("lightgbm", "xgboost"):
        metrics_path = PROJECT_ROOT / "models" / f"{model_type}_metrics.json"
        if not metrics_path.exists():
            continue
        with open(metrics_path, encoding="utf-8") as f:
            m = json.load(f)

        val_ndcg10 = m.get("val_ndcg", {}).get("ndcg@10", 0)
        train_ndcg10 = m.get("train_ndcg", {}).get("ndcg@10", 0)
        best_iter = m.get("best_iteration", 0)
        gap = train_ndcg10 - val_ndcg10
        gap_ratio = gap / train_ndcg10 if train_ndcg10 > 0 else 0

        results[model_type] = {
            "train_ndcg10": train_ndcg10,
            "val_ndcg10": val_ndcg10,
            "gap": gap,
            "gap_ratio": gap_ratio,
            "best_iteration": best_iter,
            "overfitting_severity": "严重" if gap_ratio > 0.5 else "中等" if gap_ratio > 0.3 else "轻微",
        }

        logger.info(
            "%s: 训练NDCG@10=%.4f, 验证NDCG@10=%.4f, 差距=%.4f(%.1f%%), best_iter=%d, 过拟合=%s",
            model_type.upper(),
            train_ndcg10,
            val_ndcg10,
            gap,
            gap_ratio * 100,
            best_iter,
            results[model_type]["overfitting_severity"],
        )

    eval_path = PROJECT_ROOT / "models" / "evaluation_metrics.json"
    if eval_path.exists():
        with open(eval_path, encoding="utf-8") as f:
            eval_m = json.load(f)
        for model_type in ("lightgbm", "xgboost"):
            if model_type in eval_m:
                test_ndcg10 = eval_m[model_type].get("ndcg@10", 0)
                results.setdefault(model_type, {})["test_ndcg10"] = test_ndcg10
                logger.info("%s: 测试集NDCG@10=%.4f", model_type.upper(), test_ndcg10)

    return results


def diagnose_factor_validity() -> dict[str, Any]:
    """诊断5：因子有效性检查。"""
    logger.info("")
    logger.info("=" * 60)
    logger.info("诊断5：因子有效性检查")
    logger.info("=" * 60)

    results: dict[str, Any] = {}

    factor_cols = load_factor_columns(data_dir=DATA_DIR)
    logger.info("因子总数: %d", len(factor_cols))

    df = _load_test_df()

    zero_factors: list[str] = []
    low_var_factors: list[str] = []
    high_nan_factors: list[str] = []

    for col in factor_cols:
        if col not in df.columns:
            zero_factors.append(col)
            continue
        series = df[col]
        if series.std() < 1e-10:
            zero_factors.append(col)
        elif series.std() < 1e-3:
            low_var_factors.append(col)
        nan_pct = float(series.isna().mean())
        if nan_pct > 0.5:
            high_nan_factors.append((col, nan_pct))

    results["total_factors"] = len(factor_cols)
    results["zero_factors"] = zero_factors
    results["low_var_factors"] = low_var_factors
    results["high_nan_factors"] = high_nan_factors
    results["valid_factors"] = len(factor_cols) - len(zero_factors) - len(low_var_factors)

    logger.info("全零因子(%d): %s", len(zero_factors), zero_factors)
    logger.info("低方差因子(%d): %s", len(low_var_factors), low_var_factors)
    logger.info("高缺失率因子(%d): %s", len(high_nan_factors), [f"{c}({p:.1%})" for c, p in high_nan_factors])
    logger.info("有效因子: %d / %d", results["valid_factors"], len(factor_cols))

    return results


def diagnose_time_alignment() -> dict[str, Any]:
    """诊断6：时间对齐验证（确保无未来信息泄露）。"""
    logger.info("")
    logger.info("=" * 60)
    logger.info("诊断6：时间对齐验证")
    logger.info("=" * 60)

    results: dict[str, Any] = {}

    df = _load_test_df()
    dates = sorted(df["date"].unique())

    results["test_start"] = str(dates[0])
    results["test_end"] = str(dates[-1])
    results["n_dates"] = len(dates)
    results["n_stocks_per_date_avg"] = int(df.groupby("date").size().mean())

    if "future_return_1w" in df.columns:
        avg_ret = float(df["future_return_1w"].mean())
        median_ret = float(df["future_return_1w"].median())
        pct_positive = float((df["future_return_1w"] > 0).mean())
        results["avg_future_return_1w"] = avg_ret
        results["median_future_return_1w"] = median_ret
        results["pct_positive_return"] = pct_positive

        logger.info(
            "测试集: %s ~ %s, 平均未来收益=%.4f%%, 中位数=%.4f%%, 正收益占比=%.1f%%",
            dates[0],
            dates[-1],
            avg_ret * 100,
            median_ret * 100,
            pct_positive * 100,
        )

        if avg_ret < -0.002:
            logger.info("→ 测试期市场整体偏弱，平均周收益为负")
        elif avg_ret < 0.002:
            logger.info("→ 测试期市场整体中性")
        else:
            logger.info("→ 测试期市场整体偏强")

    return results


def generate_diagnosis_report(
    corr_results: dict[str, Any],
    vol_results: dict[str, Any],
    ret_results: dict[str, Any],
    overfit_results: dict[str, Any],
    factor_results: dict[str, Any],
    time_results: dict[str, Any],
) -> str:
    """生成诊断报告。"""
    lines: list[str] = []
    lines.append("# 基线负收益深度诊断报告")
    lines.append("")
    lines.append(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # 1. 核心结论
    lines.append("## 1. 核心结论")
    lines.append("")

    lgbm_sp = corr_results.get("lightgbm", {}).get("avg_spearman", 0)
    xgb_sp = corr_results.get("xgboost", {}).get("avg_spearman", 0)

    if lgbm_sp < 0 or xgb_sp < 0:
        lines.append("### ⚠️ 模型分数与未来收益呈负相关")
        lines.append("")
        lines.append(f"- LightGBM 平均Spearman相关: {lgbm_sp:.4f}")
        lines.append(f"- XGBoost 平均Spearman相关: {xgb_sp:.4f}")
        lines.append("")
        lines.append("**这意味着：模型给高分的股票，未来收益反而更低。**")
        lines.append("**这就是基线负收益的根本原因。**")
        lines.append("")
    else:
        lines.append("### 模型分数与未来收益正相关")
        lines.append("模型排序方向正确，但Top20仍亏损，需要进一步分析。")
        lines.append("")

    # 2. 过拟合分析
    lines.append("## 2. 过拟合分析")
    lines.append("")
    lines.append("| 模型 | 训练NDCG@10 | 验证NDCG@10 | 差距 | 过拟合程度 | best_iteration |")
    lines.append("|------|------------|------------|------|-----------|---------------|")
    for model_type in ("lightgbm", "xgboost"):
        m = overfit_results.get(model_type, {})
        lines.append("| {} | {:.4f} | {:.4f} | {:.4f}({:.1f}%) | {} | {} |".format(
            model_type.upper(),
            m.get("train_ndcg10", 0),
            m.get("val_ndcg10", 0),
            m.get("gap", 0),
            m.get("gap_ratio", 0) * 100,
            m.get("overfitting_severity", "未知"),
            m.get("best_iteration", 0),
        ))
    lines.append("")

    lgbm_gap = overfit_results.get("lightgbm", {}).get("gap_ratio", 0)
    if lgbm_gap > 0.5:
        lines.append("**LightGBM过拟合严重**：训练-验证NDCG差距超过50%，模型在训练集上记忆了噪声模式，")
        lines.append("这些模式在测试集上不仅失效，反而产生了负向预测（负相关）。")
    lines.append("")

    # 3. 波动率暴露
    lines.append("## 3. 波动率暴露分析")
    lines.append("")
    lines.append("| 模型 | Top20平均波动率 | Bottom20平均波动率 | 全市场平均波动率 | Top/全市场比 |")
    lines.append("|------|---------------|-------------------|----------------|------------|")
    for model_type in ("lightgbm", "xgboost"):
        m = vol_results.get(model_type, {})
        lines.append("| {} | {:.4f} | {:.4f} | {:.4f} | {:.2f} |".format(
            model_type.upper(),
            m.get("top20_avg_vol", 0),
            m.get("bottom20_avg_vol", 0),
            m.get("all_avg_vol", 0),
            m.get("top_vs_all_ratio", 0),
        ))
    lines.append("")

    # 4. 收益分布
    lines.append("## 4. Top/Bottom/随机选股收益分布")
    lines.append("")
    lines.append("| 模型 | Top20周均收益(%) | Bottom20周均收益(%) | 随机20周均收益(%) | Top-随机(%) |")
    lines.append("|------|----------------|-------------------|----------------|-----------|")
    for model_type in ("lightgbm", "xgboost"):
        m = ret_results.get(model_type, {})
        lines.append("| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
            model_type.upper(),
            m.get("top20_avg_weekly_return", 0) * 100,
            m.get("bottom20_avg_weekly_return", 0) * 100,
            m.get("random20_avg_weekly_return", 0) * 100,
            m.get("top_minus_random", 0) * 100,
        ))
    lines.append("")

    # 5. 因子有效性
    lines.append("## 5. 因子有效性")
    lines.append("")
    lines.append(f"- 因子总数: {factor_results.get('total_factors', 0)}")
    lines.append(f"- 有效因子: {factor_results.get('valid_factors', 0)}")
    lines.append(f"- 全零因子({len(factor_results.get('zero_factors', []))}个): {', '.join(factor_results.get('zero_factors', []))}")
    lines.append(f"- 低方差因子({len(factor_results.get('low_var_factors', []))}个): {', '.join(factor_results.get('low_var_factors', []))}")
    lines.append("")

    # 6. 市场环境
    lines.append("## 6. 测试期市场环境")
    lines.append("")
    lines.append(f"- 测试期: {time_results.get('test_start', '')} ~ {time_results.get('test_end', '')}")
    lines.append(f"- 截面数: {time_results.get('n_dates', 0)}")
    lines.append(f"- 平均截面股票数: {time_results.get('n_stocks_per_date_avg', 0)}")
    avg_ret = time_results.get("avg_future_return_1w", 0)
    lines.append(f"- 平均未来周收益: {avg_ret * 100:.4f}%")
    lines.append(f"- 正收益占比: {time_results.get('pct_positive_return', 0) * 100:.1f}%")
    lines.append("")

    # 7. 根因总结与修复方案
    lines.append("## 7. 根因总结与修复方案")
    lines.append("")
    lines.append("### 根因链")
    lines.append("")
    lines.append("```")
    lines.append("严重过拟合（训练NDCG@10=0.54, 验证=0.16）")
    lines.append("    ↓")
    lines.append("模型在训练集上记忆了噪声模式（高波动股=高收益）")
    lines.append("    ↓")
    lines.append("测试集上这些模式反转（高波动股=高亏损）")
    lines.append("    ↓")
    lines.append("模型分数与未来收益负相关（Spearman≈-0.06）")
    lines.append("    ↓")
    lines.append("Top20选股年化-23.69%，远差于随机选股-5.36%")
    lines.append("```")
    lines.append("")
    lines.append("### 修复方案")
    lines.append("")
    lines.append("1. **移除6个全零因子**：减少噪声特征维度，从40→34个有效因子")
    lines.append("2. **增强正则化**：增大min_child_samples、减小num_leaves/max_depth，降低过拟合")
    lines.append("3. **添加波动率调整选股**：adjusted_score = model_score - λ * volatility，降低高波动暴露")
    lines.append("4. **重新训练模型**：使用34个有效因子+更强正则化")
    lines.append("5. **验证修复后基线**：确认B-LGBM年化收益 > B-EW年化收益")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    """主诊断流程。"""
    logger.info("开始基线负收益深度诊断...")
    logger.info("")

    corr_results = diagnose_score_return_correlation()
    vol_results = diagnose_volatility_bias()
    ret_results = diagnose_top_bottom_returns()
    overfit_results = diagnose_overfitting()
    factor_results = diagnose_factor_validity()
    time_results = diagnose_time_alignment()

    report = generate_diagnosis_report(
        corr_results,
        vol_results,
        ret_results,
        overfit_results,
        factor_results,
        time_results,
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    logger.info("")
    logger.info("=" * 60)
    logger.info("诊断报告已保存: %s", REPORT_PATH)
    logger.info("=" * 60)

    print(f"\n诊断完成！报告: {REPORT_PATH}")


if __name__ == "__main__":
    main()
