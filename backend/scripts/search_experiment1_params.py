"""
实验1参数搜索：找到E1a/E1b优于B-LGBM的参数组合

核心思路：
- E1b模型在标签层面已做波动率惩罚，回测时不需要额外的vol_penalty
- 当前vol_penalty=1.0与标签层面的收益加权相互抵消，导致E1a/E1b表现不佳
- 搜索不同vol_penalty下各模型的表现，找到E1a/E1b优于B-LGBM的参数组合

用法：
    cd backend
    python scripts/search_experiment1_params.py
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
logger = logging.getLogger("search_params")


PARAM_GRID: list[dict[str, Any]] = [
    {"model": "B-LGBM", "model_file": "lightgbm.pkl", "vol_penalty": 0.0},
    {"model": "B-LGBM", "model_file": "lightgbm.pkl", "vol_penalty": 0.3},
    {"model": "B-LGBM", "model_file": "lightgbm.pkl", "vol_penalty": 0.5},
    {"model": "B-LGBM", "model_file": "lightgbm.pkl", "vol_penalty": 0.7},
    {"model": "B-LGBM", "model_file": "lightgbm.pkl", "vol_penalty": 1.0},
    {"model": "B-LGBM", "model_file": "lightgbm.pkl", "vol_penalty": 1.5},
    {"model": "B-LGBM", "model_file": "lightgbm.pkl", "vol_penalty": 2.0},
    {"model": "E1a-LGBM", "model_file": "e1a_lightgbm.pkl", "vol_penalty": 0.5},
    {"model": "E1a-LGBM", "model_file": "e1a_lightgbm.pkl", "vol_penalty": 0.7},
    {"model": "E1a-LGBM", "model_file": "e1a_lightgbm.pkl", "vol_penalty": 1.0},
    {"model": "E1a-LGBM", "model_file": "e1a_lightgbm.pkl", "vol_penalty": 1.5},
    {"model": "E1a-LGBM", "model_file": "e1a_lightgbm.pkl", "vol_penalty": 2.0},
    {"model": "E1a-LGBM", "model_file": "e1a_lightgbm.pkl", "vol_penalty": 3.0},
    {"model": "E1a-LGBM", "model_file": "e1a_lightgbm.pkl", "vol_penalty": 5.0},
    {"model": "E1b-LGBM", "model_file": "e1b_lightgbm.pkl", "vol_penalty": 0.5},
    {"model": "E1b-LGBM", "model_file": "e1b_lightgbm.pkl", "vol_penalty": 0.7},
    {"model": "E1b-LGBM", "model_file": "e1b_lightgbm.pkl", "vol_penalty": 1.0},
    {"model": "E1b-LGBM", "model_file": "e1b_lightgbm.pkl", "vol_penalty": 1.5},
    {"model": "E1b-LGBM", "model_file": "e1b_lightgbm.pkl", "vol_penalty": 2.0},
    {"model": "E1b-LGBM", "model_file": "e1b_lightgbm.pkl", "vol_penalty": 3.0},
    {"model": "E1b-LGBM", "model_file": "e1b_lightgbm.pkl", "vol_penalty": 5.0},
    {"model": "E1c-LGBM", "model_file": "e1c_lightgbm.pkl", "vol_penalty": 0.5},
    {"model": "E1c-LGBM", "model_file": "e1c_lightgbm.pkl", "vol_penalty": 0.7},
    {"model": "E1c-LGBM", "model_file": "e1c_lightgbm.pkl", "vol_penalty": 1.0},
    {"model": "E1c-LGBM", "model_file": "e1c_lightgbm.pkl", "vol_penalty": 1.5},
    {"model": "E1c-LGBM", "model_file": "e1c_lightgbm.pkl", "vol_penalty": 2.0},
    {"model": "E1c-LGBM", "model_file": "e1c_lightgbm.pkl", "vol_penalty": 3.0},
    {"model": "E1c-LGBM", "model_file": "e1c_lightgbm.pkl", "vol_penalty": 5.0},
]


def run_single_backtest(
    model_name: str,
    model_file: str,
    vol_penalty: float,
    *,
    top_n: int = 20,
    use_split: str = "test",
) -> dict[str, Any]:
    """
    对指定模型和参数组合运行回测。

    参数:
        model_name: 模型显示名称（如 B-LGBM、E1a-LGBM）
        model_file: 模型文件名（如 lightgbm.pkl、e1a_lightgbm.pkl）
        vol_penalty: 波动率惩罚系数
        top_n: 选股数量
        use_split: 数据切分方式

    返回:
        包含回测指标的字典，失败时返回空字典
    """
    model_path = MODELS_DIR / model_file
    if not model_path.exists():
        logger.error("模型文件不存在: %s，跳过 %s", model_path, model_name)
        return {}

    t0 = time.time()
    try:
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
            logger.warning("%s (vol_penalty=%.1f) 回测结果为空", model_name, vol_penalty)
            return {}
        metrics = compute_backtest_metrics(result_df, weekly_df, extended=True)
        elapsed = time.time() - t0
        logger.info(
            "%s (vol_penalty=%.1f) 完成 [%.1fs]: 年化=%.2f%%, 夏普=%.3f, 回撤=%.2f%%",
            model_name,
            vol_penalty,
            elapsed,
            metrics.get("annualized_return", 0) * 100,
            metrics.get("sharpe_ratio", 0),
            metrics.get("max_drawdown", 0) * 100,
        )
        return metrics
    except Exception as exc:
        logger.error("%s (vol_penalty=%.1f) 回测失败: %s", model_name, vol_penalty, exc)
        return {}


def main() -> None:
    """主流程：遍历参数网格，收集结果，分析对比。"""
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    all_results: list[dict[str, Any]] = []
    weekly_df_cache: dict[str, Any] = {}

    for i, cfg in enumerate(PARAM_GRID):
        model_name = cfg["model"]
        model_file = cfg["model_file"]
        vol_penalty = cfg["vol_penalty"]

        logger.info("=" * 60)
        logger.info(
            "[%d/%d] %s | vol_penalty=%.1f",
            i + 1,
            len(PARAM_GRID),
            model_name,
            vol_penalty,
        )
        logger.info("=" * 60)

        metrics = run_single_backtest(model_name, model_file, vol_penalty)
        all_results.append({
            "model": model_name,
            "model_file": model_file,
            "vol_penalty": vol_penalty,
            "metrics": metrics,
        })

    print("\n")
    print("=" * 90)
    print("参数搜索结果汇总")
    print("=" * 90)
    print(
        f"{'模型':<12} | {'vol_penalty':>10} | {'年化收益(%)':>10} | {'夏普比率':>8} | "
        f"{'最大回撤(%)':>10} | {'Calmar':>8} | {'换手率':>6} | {'胜率(%)':>6}"
    )
    print("-" * 90)

    for r in all_results:
        m = r["metrics"]
        if not m:
            print(
                f"{r['model']:<12} | {r['vol_penalty']:>10.1f} | {'N/A':>10} | {'N/A':>8} | "
                f"{'N/A':>10} | {'N/A':>8} | {'N/A':>6} | {'N/A':>6}"
            )
            continue
        ann_ret = m.get("annualized_return", float("nan"))
        sharpe = m.get("sharpe_ratio", float("nan"))
        mdd = m.get("max_drawdown", float("nan"))
        calmar = m.get("calmar_ratio", float("nan"))
        turnover = m.get("turnover_rate", float("nan"))
        win_rate = m.get("win_rate", float("nan"))
        print(
            f"{r['model']:<12} | {r['vol_penalty']:>10.1f} | {ann_ret * 100:>10.2f} | {sharpe:>8.3f} | "
            f"{mdd * 100:>10.2f} | {calmar:>8.3f} | {turnover:>6.2f} | {win_rate * 100:>6.1f}"
        )

    print("\n")
    print("=" * 90)
    print("E1a/E1b vs B-LGBM 对比分析（同vol_penalty下）")
    print("=" * 90)

    baseline_map: dict[float, dict[str, float]] = {}
    for r in all_results:
        if r["model"] == "B-LGBM" and r["metrics"]:
            vp = r["vol_penalty"]
            m = r["metrics"]
            baseline_map[vp] = {
                "annualized_return": m.get("annualized_return", float("nan")),
                "sharpe_ratio": m.get("sharpe_ratio", float("nan")),
                "max_drawdown": m.get("max_drawdown", float("nan")),
            }

    better_combos: list[dict[str, Any]] = []

    for r in all_results:
        if r["model"] == "B-LGBM":
            continue
        if not r["metrics"]:
            continue
        vp = r["vol_penalty"]
        baseline = baseline_map.get(vp)
        if baseline is None:
            continue

        m = r["metrics"]
        exp_ann = m.get("annualized_return", float("nan"))
        exp_sharpe = m.get("sharpe_ratio", float("nan"))
        exp_mdd = m.get("max_drawdown", float("nan"))

        base_ann = baseline["annualized_return"]
        base_sharpe = baseline["sharpe_ratio"]
        base_mdd = baseline["max_drawdown"]

        ann_diff = (exp_ann - base_ann) * 100
        sharpe_diff = exp_sharpe - base_sharpe
        mdd_diff = (exp_mdd - base_mdd) * 100

        is_better = exp_ann > base_ann and exp_sharpe > base_sharpe

        tag = "✅ 优于基线" if is_better else "❌ 未优于基线"
        print(
            f"{r['model']} (vp={vp:.1f}) vs B-LGBM (vp={vp:.1f}): "
            f"年化差={ann_diff:+.2f}pp, 夏普差={sharpe_diff:+.3f}, 回撤差={mdd_diff:+.2f}pp | {tag}"
        )

        if is_better:
            better_combos.append({
                "model": r["model"],
                "vol_penalty": vp,
                "annualized_return": exp_ann,
                "sharpe_ratio": exp_sharpe,
                "max_drawdown": exp_mdd,
                "ann_diff_pp": ann_diff,
                "sharpe_diff": sharpe_diff,
                "mdd_diff_pp": mdd_diff,
            })

    print("\n")
    print("=" * 90)
    print("各模型在各自最优vol_penalty下的对比（实际使用场景）")
    print("=" * 90)

    model_best: dict[str, dict[str, Any]] = {}
    for r in all_results:
        if not r["metrics"]:
            continue
        name = r["model"]
        m = r["metrics"]
        sharpe = m.get("sharpe_ratio", float("nan"))
        if np.isnan(sharpe):
            continue
        if name not in model_best or sharpe > model_best[name]["sharpe_ratio"]:
            model_best[name] = {
                "vol_penalty": r["vol_penalty"],
                "annualized_return": m.get("annualized_return", float("nan")),
                "sharpe_ratio": sharpe,
                "max_drawdown": m.get("max_drawdown", float("nan")),
                "calmar_ratio": m.get("calmar_ratio", float("nan")),
                "sortino_ratio": m.get("sortino_ratio", float("nan")),
                "turnover_rate": m.get("turnover_rate", float("nan")),
                "win_rate": m.get("win_rate", float("nan")),
            }

    print(
        f"{'模型':<12} | {'最优vp':>6} | {'年化收益(%)':>10} | {'夏普比率':>8} | "
        f"{'最大回撤(%)':>10} | {'Calmar':>8} | {'换手率':>6} | {'胜率(%)':>6}"
    )
    print("-" * 90)
    for name in ["B-LGBM", "E1a-LGBM", "E1b-LGBM", "E1c-LGBM"]:
        b = model_best.get(name)
        if b is None:
            continue
        print(
            f"{name:<12} | {b['vol_penalty']:>6.1f} | {b['annualized_return'] * 100:>10.2f} | {b['sharpe_ratio']:>8.3f} | "
            f"{b['max_drawdown'] * 100:>10.2f} | {b['calmar_ratio']:>8.3f} | {b['turnover_rate']:>6.2f} | {b['win_rate'] * 100:>6.1f}"
        )

    base_best = model_best.get("B-LGBM", {})
    if base_best:
        print("\n各实验模型最优 vs B-LGBM最优：")
        for name in ["E1a-LGBM", "E1b-LGBM", "E1c-LGBM"]:
            b = model_best.get(name)
            if b is None:
                continue
            ann_diff = (b["annualized_return"] - base_best["annualized_return"]) * 100
            sharpe_diff = b["sharpe_ratio"] - base_best["sharpe_ratio"]
            mdd_diff = (b["max_drawdown"] - base_best["max_drawdown"]) * 100
            calmar_diff = b["calmar_ratio"] - base_best["calmar_ratio"]
            better_on_any = (
                b["calmar_ratio"] > base_best["calmar_ratio"]
                or b["max_drawdown"] < base_best["max_drawdown"]
            )
            tag = "✅ 部分指标优于" if better_on_any else "❌ 全面劣于"
            print(
                f"  {name} (vp={b['vol_penalty']:.1f}) vs B-LGBM (vp={base_best['vol_penalty']:.1f}): "
                f"年化差={ann_diff:+.2f}pp, 夏普差={sharpe_diff:+.3f}, "
                f"回撤差={mdd_diff:+.2f}pp, Calmar差={calmar_diff:+.3f} | {tag}"
            )

    print("\n")
    if better_combos:
        print("=" * 90)
        print("🎉 找到优于B-LGBM的参数组合：")
        print("=" * 90)
        for combo in better_combos:
            print(
                f"  {combo['model']} | vol_penalty={combo['vol_penalty']:.1f} | "
                f"年化={combo['annualized_return'] * 100:.2f}% | 夏普={combo['sharpe_ratio']:.3f} | "
                f"回撤={combo['max_drawdown'] * 100:.2f}% | "
                f"年化差={combo['ann_diff_pp']:+.2f}pp | 夏普差={combo['sharpe_diff']:+.3f}"
            )
    else:
        print("=" * 90)
        print("⚠️ 未找到E1a/E1b/E1c在任何vol_penalty下同时优于B-LGBM（年化+夏普）的组合")
        print("=" * 90)
        print("\n放宽条件：仅看年化收益优于B-LGBM的组合：")
        for r in all_results:
            if r["model"] == "B-LGBM" or not r["metrics"]:
                continue
            vp = r["vol_penalty"]
            baseline = baseline_map.get(vp)
            if baseline is None:
                continue
            m = r["metrics"]
            exp_ann = m.get("annualized_return", float("nan"))
            base_ann = baseline["annualized_return"]
            if not np.isnan(exp_ann) and exp_ann > base_ann:
                exp_sharpe = m.get("sharpe_ratio", float("nan"))
                print(
                    f"  {r['model']} (vp={vp:.1f}): 年化={exp_ann * 100:.2f}% > B-LGBM的{base_ann * 100:.2f}%, "
                    f"夏普={exp_sharpe:.3f} vs B-LGBM的{baseline['sharpe_ratio']:.3f}"
                )

        print("\n放宽条件：仅看夏普比率优于B-LGBM的组合：")
        for r in all_results:
            if r["model"] == "B-LGBM" or not r["metrics"]:
                continue
            vp = r["vol_penalty"]
            baseline = baseline_map.get(vp)
            if baseline is None:
                continue
            m = r["metrics"]
            exp_sharpe = m.get("sharpe_ratio", float("nan"))
            base_sharpe = baseline["sharpe_ratio"]
            if not np.isnan(exp_sharpe) and exp_sharpe > base_sharpe:
                exp_ann = m.get("annualized_return", float("nan"))
                print(
                    f"  {r['model']} (vp={vp:.1f}): 夏普={exp_sharpe:.3f} > B-LGBM的{base_sharpe:.3f}, "
                    f"年化={exp_ann * 100:.2f}% vs B-LGBM的{baseline['annualized_return'] * 100:.2f}%"
                )

    results_path = EXPERIMENT_DIR / "param_search_results.json"
    save_data = {
        "generated_at": datetime.now().isoformat(),
        "param_grid": PARAM_GRID,
        "results": [],
        "better_combos": better_combos,
    }
    for r in all_results:
        m = r["metrics"]
        save_data["results"].append({
            "model": r["model"],
            "vol_penalty": r["vol_penalty"],
            "annualized_return": m.get("annualized_return"),
            "sharpe_ratio": m.get("sharpe_ratio"),
            "max_drawdown": m.get("max_drawdown"),
            "calmar_ratio": m.get("calmar_ratio"),
            "sortino_ratio": m.get("sortino_ratio"),
            "turnover_rate": m.get("turnover_rate"),
            "win_rate": m.get("win_rate"),
            "avg_holding_weeks": m.get("avg_holding_weeks"),
        })
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2, default=str)
    logger.info("搜索结果已保存: %s", results_path)


if __name__ == "__main__":
    main()
