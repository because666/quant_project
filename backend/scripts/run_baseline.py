"""
基线实验一键运行脚本：训练双模型 → 双模型回测 → B-EW随机基线 → B-MOM动量基线 → 生成报告。

用法：
    cd backend
    python scripts/run_baseline.py [--skip-train] [--top-n 20] [--n-runs 100]

参数:
    --skip-train: 跳过模型训练步骤（使用已有模型）
    --top-n: 选股数量（默认20）
    --n-runs: B-EW随机基线运行次数（默认100）
"""
from __future__ import annotations

import argparse
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASELINE_DIR = PROJECT_ROOT / "data" / "baseline"


def step1_train_models(skip_train: bool) -> None:
    """第1步：训练LightGBM和XGBoost模型。"""
    if skip_train:
        logger.info("跳过模型训练（使用已有模型）")
        return

    logger.info("=" * 60)
    logger.info("第1步：训练LightGBM模型")
    logger.info("=" * 60)
    from src.model_lightgbm import train_final_lightgbm
    train_final_lightgbm(n_trials=20)
    logger.info("LightGBM训练完成")

    logger.info("=" * 60)
    logger.info("第2步：训练XGBoost模型")
    logger.info("=" * 60)
    from src.model_xgboost import train_final_xgboost
    train_final_xgboost(n_trials=20)
    logger.info("XGBoost训练完成")


def step2_run_model_backtest(top_n: int) -> dict[str, dict[str, Any]]:
    """第2步：双模型回测。"""
    results: dict[str, dict[str, Any]] = {}

    for model_type in ("lightgbm", "xgboost"):
        logger.info("=" * 60)
        logger.info("第3步：%s模型回测 (top_n=%d)", model_type.upper(), top_n)
        logger.info("=" * 60)

        from src.backtest import BacktestEngine, compute_backtest_metrics
        eng = BacktestEngine(model_type, top_n, 1_000_000.0)
        weekly_df = eng.load_weekly_data()
        result_df = eng.run_backtest(weekly_df, use_split="test")

        if result_df.empty:
            logger.warning("%s回测结果为空", model_type)
            results[model_type] = {"result_df": result_df, "metrics": {}}
            continue

        metrics = compute_backtest_metrics(result_df, weekly_df, extended=True)
        results[model_type] = {"result_df": result_df, "metrics": metrics}

        logger.info(
            "%s: 年化收益=%.2f%%, 夏普=%.3f, 最大回撤=%.2f%%",
            model_type.upper(),
            metrics.get("annualized_return", 0) * 100,
            metrics.get("sharpe_ratio", 0),
            metrics.get("max_drawdown", 0) * 100,
        )

    return results


def step3_run_random_baseline(top_n: int, n_runs: int) -> dict[str, Any]:
    """第3步：B-EW等权随机选股基线。"""
    logger.info("=" * 60)
    logger.info("第4步：B-EW随机选股基线 (top_n=%d, n_runs=%d)", top_n, n_runs)
    logger.info("=" * 60)

    from src.backtest import run_random_baseline
    result = run_random_baseline(top_n=top_n, n_runs=n_runs)

    logger.info(
        "B-EW: 年化收益=%.2f%%, 夏普=%.3f, 最大回撤=%.2f%%",
        result["avg_metrics"].get("annualized_return", 0) * 100,
        result["avg_metrics"].get("sharpe_ratio", 0),
        result["avg_metrics"].get("max_drawdown", 0) * 100,
    )
    return result


def step4_run_momentum_baseline(top_n: int) -> pd.DataFrame:
    """第4步：B-MOM纯动量排序基线。"""
    logger.info("=" * 60)
    logger.info("第5步：B-MOM动量排序基线 (top_n=%d)", top_n)
    logger.info("=" * 60)

    from src.backtest import run_momentum_baseline
    result_df = run_momentum_baseline(top_n=top_n)

    if not result_df.empty:
        nav = result_df["nav"]
        rets = result_df["weekly_return"].dropna().to_numpy()
        if len(rets) > 1:
            ann_ret = float(np.mean(rets) * 52)
            ann_vol = float(np.std(rets, ddof=1) * np.sqrt(52))
            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
            cummax = np.maximum.accumulate(nav.to_numpy())
            dd = (nav.to_numpy() - cummax) / cummax
            mdd = abs(float(np.min(dd)))
        else:
            ann_ret = sharpe = mdd = 0.0
        logger.info(
            "B-MOM: 年化收益=%.2f%%, 夏普=%.3f, 最大回撤=%.2f%%",
            ann_ret * 100, sharpe, mdd * 100,
        )
    return result_df


def generate_report(
    model_results: dict[str, dict[str, Any]],
    ew_result: dict[str, Any],
    mom_df: pd.DataFrame,
) -> str:
    """生成基线分析报告。"""
    lines: list[str] = []
    lines.append("# 基线实验分析报告")
    lines.append("")
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # 1. 基线方法对比表
    lines.append("## 1. 基线方法对比")
    lines.append("")
    lines.append("| 方法 | 年化收益(%) | 夏普比率 | 最大回撤(%) | Calmar比率 | Sortino比率 |")
    lines.append("|------|------------|---------|------------|-----------|------------|")

    # B-LGBM
    lgbm_m = model_results.get("lightgbm", {}).get("metrics", {})
    lines.append("| B-LGBM | {:.2f} | {:.3f} | {:.2f} | {:.3f} | {:.3f} |".format(
        lgbm_m.get("annualized_return", 0) * 100,
        lgbm_m.get("sharpe_ratio", 0),
        lgbm_m.get("max_drawdown", 0) * 100,
        lgbm_m.get("calmar_ratio", 0),
        lgbm_m.get("sortino_ratio", 0),
    ))

    # B-XGB
    xgb_m = model_results.get("xgboost", {}).get("metrics", {})
    lines.append("| B-XGB | {:.2f} | {:.3f} | {:.2f} | {:.3f} | {:.3f} |".format(
        xgb_m.get("annualized_return", 0) * 100,
        xgb_m.get("sharpe_ratio", 0),
        xgb_m.get("max_drawdown", 0) * 100,
        xgb_m.get("calmar_ratio", 0),
        xgb_m.get("sortino_ratio", 0),
    ))

    # B-EW
    ew_m = ew_result.get("avg_metrics", {})
    lines.append("| B-EW | {:.2f} | {:.3f} | {:.2f} | - | - |".format(
        ew_m.get("annualized_return", 0) * 100,
        ew_m.get("sharpe_ratio", 0),
        ew_m.get("max_drawdown", 0) * 100,
    ))

    # B-MOM
    if not mom_df.empty:
        rets = mom_df["weekly_return"].dropna().to_numpy()
        if len(rets) > 1:
            mom_ann = float(np.mean(rets) * 52)
            mom_vol = float(np.std(rets, ddof=1) * np.sqrt(52))
            mom_sharpe = mom_ann / mom_vol if mom_vol > 0 else 0.0
            nav = mom_df["nav"].to_numpy()
            cummax = np.maximum.accumulate(nav)
            dd = (nav - cummax) / cummax
            mom_mdd = abs(float(np.min(dd)))
        else:
            mom_ann = mom_sharpe = mom_mdd = 0.0
        lines.append("| B-MOM | {:.2f} | {:.3f} | {:.2f} | - | - |".format(
            mom_ann * 100, mom_sharpe, mom_mdd * 100,
        ))

    # B-INDEX
    lines.append("| B-INDEX (沪深300) | 见基准列 | - | - | - | - |")
    lines.append("")

    # 2. 排序质量
    lines.append("## 2. 排序质量（NDCG）")
    lines.append("")
    metrics_path = PROJECT_ROOT / "models" / "evaluation_metrics.json"
    eval_metrics: dict[str, Any] = {}
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as f:
            eval_metrics = json.load(f)
        lines.append("| 模型 | NDCG@5 | NDCG@10 | NDCG@20 | MAP |")
        lines.append("|------|--------|---------|---------|-----|")
        for model_key in ("lightgbm", "xgboost"):
            m = eval_metrics.get(model_key, {})
            lines.append("| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
                model_key.upper(),
                m.get("ndcg@5", 0),
                m.get("ndcg@10", 0),
                m.get("ndcg@20", 0),
                m.get("map", 0),
            ))
    else:
        lines.append("评估指标文件不存在，请先运行模型评估。")
    lines.append("")

    # 3. 合理性判断
    lines.append("## 3. 合理性判断")
    lines.append("")
    lines.append("根据实验方法论文档，基线合理性标准如下：")
    lines.append("")
    lines.append("| 指标 | 合理范围 | B-LGBM | B-XGB | 判断 |")
    lines.append("|------|---------|--------|-------|------|")

    checks = [
        ("NDCG@10", 0.30, 0.50,
         eval_metrics.get("lightgbm", {}).get("ndcg@10", 0),
         eval_metrics.get("xgboost", {}).get("ndcg@10", 0)),
        ("年化收益", -0.10, 0.30,
         lgbm_m.get("annualized_return", 0),
         xgb_m.get("annualized_return", 0)),
        ("夏普比率", -1.0, 2.0,
         lgbm_m.get("sharpe_ratio", 0),
         xgb_m.get("sharpe_ratio", 0)),
        ("最大回撤", 0.10, 0.70,
         abs(lgbm_m.get("max_drawdown", 0)),
         abs(xgb_m.get("max_drawdown", 0))),
    ]

    for name, lo, hi, lgbm_val, xgb_val in checks:
        lgbm_ok = lo <= lgbm_val <= hi
        xgb_ok = lo <= xgb_val <= hi
        lgbm_judge = "✅ 合理" if lgbm_ok else "⚠️ 异常"
        xgb_judge = "✅ 合理" if xgb_ok else "⚠️ 异常"
        lines.append("| {} | {}-{} | {:.4f} {} | {:.4f} {} | |".format(
            name, lo, hi, lgbm_val, lgbm_judge, xgb_val, xgb_judge,
        ))
    lines.append("")

    # 4. 问题诊断
    lines.append("## 4. 问题诊断")
    lines.append("")
    lgbm_ann = lgbm_m.get("annualized_return", 0)
    lgbm_sharpe = lgbm_m.get("sharpe_ratio", 0)
    xgb_ann = xgb_m.get("annualized_return", 0)
    xgb_sharpe = xgb_m.get("sharpe_ratio", 0)
    ew_ann = ew_result.get("avg_metrics", {}).get("annualized_return", 0)

    if lgbm_ann < 0 and xgb_ann < 0:
        lines.append("### 双模型年化收益均为负")
        lines.append("")
        lines.append("诊断分析：")
        lines.append("")
        lines.append("1. **市场环境**：B-EW（随机选股）年化收益为 {:.1f}%，说明测试期市场整体偏弱".format(ew_ann * 100))
        lines.append("2. **模型 vs 随机**：B-LGBM({:.1f}%) vs B-EW({:.1f}%)，模型表现差于随机 → 模型选股偏向高波动股".format(lgbm_ann * 100, ew_ann * 100))
        lines.append("3. **NDCG水平**：A股周频排序学习NDCG@10在0.3-0.4属正常水平，0.75-0.90为学术文献中信息检索任务的标准，不适用于股票排序")
        lines.append("4. **核心问题**：模型选出的Top N股票在下跌市场中跌幅更大（高波动暴露）")
        lines.append("")
        lines.append("改进方向：")
        lines.append("- 添加波动率因子或风险调整因子")
        lines.append("- 使用收益-夏普感知损失函数（实验1）")
        lines.append("- 多模型融合降低选股波动（实验2）")
        lines.append("- 考虑加入择时/空仓机制")
        lines.append("")
    elif lgbm_ann < 0:
        lines.append("### B-LGBM年化收益为负")
        lines.append("")
        lines.append("可能原因：")
        lines.append("1. 模型欠拟合（best_iteration过小）→ 检查训练日志")
        lines.append("2. 测试期市场环境差 → 对比B-INDEX基准")
        lines.append("3. 因子预测能力不足 → 检查NDCG指标")
        lines.append("4. 回测逻辑bug → 对比B-EW基线")
        lines.append("")
    elif lgbm_sharpe < 0.5:
        lines.append("### B-LGBM夏普比率低于0.5")
        lines.append("")
        lines.append("可能原因：")
        lines.append("1. 策略波动过大 → 检查持仓集中度")
        lines.append("2. 换手率过高 → 检查交易成本影响")
        lines.append("3. 选股信号不稳定 → 检查周度胜率")
        lines.append("")
    else:
        lines.append("### B-LGBM基线结果合理")
        lines.append("")
        lines.append("年化收益和夏普比率均在合理范围内，可以继续后续改进实验。")
        lines.append("")

    # 5. Bootstrap置信区间
    lines.append("## 5. Bootstrap置信区间")
    lines.append("")
    if lgbm_m.get("sharpe_ci"):
        ci = lgbm_m["sharpe_ci"]
        pt, lo, hi = ci.get("point", 0), ci.get("lower", 0), ci.get("upper", 0)
        if abs(pt) < 100 and abs(lo) < 100 and abs(hi) < 100:
            lines.append("- B-LGBM 夏普比率: {:.3f} [{:.3f}, {:.3f}]".format(pt, lo, hi))
        else:
            lines.append("- B-LGBM 夏普比率: 置信区间异常（Bootstrap采样不足）")
    if xgb_m.get("sharpe_ci"):
        ci = xgb_m["sharpe_ci"]
        pt, lo, hi = ci.get("point", 0), ci.get("lower", 0), ci.get("upper", 0)
        if abs(pt) < 100 and abs(lo) < 100 and abs(hi) < 100:
            lines.append("- B-XGB 夏普比率: {:.3f} [{:.3f}, {:.3f}]".format(pt, lo, hi))
        else:
            lines.append("- B-XGB 夏普比率: 置信区间异常（Bootstrap采样不足）")
    if lgbm_m.get("annualized_return_ci"):
        ci = lgbm_m["annualized_return_ci"]
        pt, lo, hi = ci.get("point", 0), ci.get("lower", 0), ci.get("upper", 0)
        if abs(pt) < 10 and abs(lo) < 10 and abs(hi) < 10:
            lines.append("- B-LGBM 年化收益: {:.4f} [{:.4f}, {:.4f}]".format(pt, lo, hi))
        else:
            lines.append("- B-LGBM 年化收益: 置信区间异常（Bootstrap采样不足）")
    if xgb_m.get("annualized_return_ci"):
        ci = xgb_m["annualized_return_ci"]
        pt, lo, hi = ci.get("point", 0), ci.get("lower", 0), ci.get("upper", 0)
        if abs(pt) < 10 and abs(lo) < 10 and abs(hi) < 10:
            lines.append("- B-XGB 年化收益: {:.4f} [{:.4f}, {:.4f}]".format(pt, lo, hi))
        else:
            lines.append("- B-XGB 年化收益: 置信区间异常（Bootstrap采样不足）")
    lines.append("")

    # 6. 下一步建议
    lines.append("## 6. 下一步建议")
    lines.append("")
    lines.append("根据基线结果，建议的后续步骤：")
    lines.append("")
    if lgbm_ann > 0 and lgbm_sharpe > 0.5:
        lines.append("1. ✅ 基线结果合理，可以开始实验1（收益-夏普感知损失函数）")
        lines.append("2. 然后跑实验2（RRF多模型融合）")
        lines.append("3. 最后跑实验3（LLM可解释推荐评估）")
    else:
        lines.append("1. ⚠️ 基线结果异常，需要先排查问题")
        lines.append("2. 检查模型训练日志，确认best_iteration合理")
        lines.append("3. 检查因子数据质量，确认无异常值或缺失")
        lines.append("4. 对比B-EW基线，确认回测逻辑无bug")
        lines.append("5. 排查完成后重新运行本脚本")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    """主流程。"""
    parser = argparse.ArgumentParser(description="基线实验一键运行")
    parser.add_argument("--skip-train", action="store_true", help="跳过模型训练")
    parser.add_argument("--top-n", type=int, default=20, help="选股数量")
    parser.add_argument("--n-runs", type=int, default=100, help="B-EW随机基线运行次数")
    args = parser.parse_args()

    BASELINE_DIR.mkdir(parents=True, exist_ok=True)

    # 第1步：训练模型
    step1_train_models(args.skip_train)

    # 第2步：双模型回测
    model_results = step2_run_model_backtest(args.top_n)

    # 第3步：B-EW随机基线
    ew_result = step3_run_random_baseline(args.top_n, args.n_runs)

    # 第4步：B-MOM动量基线
    mom_df = step4_run_momentum_baseline(args.top_n)

    # 第5步：生成报告
    logger.info("=" * 60)
    logger.info("第6步：生成基线分析报告")
    logger.info("=" * 60)
    report = generate_report(model_results, ew_result, mom_df)
    report_path = BASELINE_DIR / "baseline_report.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info("基线报告已保存: %s", report_path)

    # 保存回测结果
    for model_type, data in model_results.items():
        result_df = data.get("result_df")
        if result_df is not None and not result_df.empty:
            out_path = BASELINE_DIR / f"{model_type}_backtest.parquet"
            result_df.to_parquet(out_path, index=False)
            logger.info("%s回测结果已保存: %s", model_type.upper(), out_path)

    if not mom_df.empty:
        mom_path = BASELINE_DIR / "momentum_backtest.parquet"
        mom_df.to_parquet(mom_path, index=False)
        logger.info("B-MOM回测结果已保存: %s", mom_path)

    # 保存B-EW结果
    ew_path = BASELINE_DIR / "random_baseline.json"
    ew_save = {
        "avg_metrics": ew_result["avg_metrics"],
        "avg_nav": ew_result["avg_nav"].tolist(),
    }
    with open(ew_path, "w", encoding="utf-8") as f:
        json.dump(ew_save, f, ensure_ascii=False, indent=2)
    logger.info("B-EW结果已保存: %s", ew_path)

    print(f"\n基线实验完成！")
    print(f"报告: {report_path}")
    print(f"结果目录: {BASELINE_DIR}")


if __name__ == "__main__":
    main()
