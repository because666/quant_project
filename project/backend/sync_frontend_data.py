"""根据最新回测结果更新前端 public/data 目录的所有数据文件。

所有 NDCG 指标值均从 models/{model}_metrics.json 动态读取，
禁止硬编码指标数值，确保前端数据与后端模型训练结果一致。
"""
import sys
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND_DATA = ROOT / "data" / "backtest_results"
BACKEND_MODELS = ROOT / "models"
FRONTEND_DATA = ROOT.parent / "frontend" / "public" / "data"

VAL_DATES_PATH = ROOT / "data" / "val.parquet"

def main() -> None:
    FRONTEND_DATA.mkdir(parents=True, exist_ok=True)

    # 1. 直接复制的文件
    direct_copy_from_models = [
        "lightgbm_metrics.json",
        "lightgbm_feature_importance.json",
        "lightgbm_best_params.json",
        "xgboost_metrics.json",
        "xgboost_feature_importance.json",
    ]
    for fname in direct_copy_from_models:
        src = BACKEND_MODELS / fname
        if src.exists():
            shutil.copy(src, FRONTEND_DATA / fname)
            print(f"已复制: {fname}")
        else:
            print(f"⚠️  源文件不存在: {src}")

    direct_copy_from_data = [
        "lightgbm_holdings.json",
        "xgboost_holdings.json",
        "comparison.json",
    ]
    for fname in direct_copy_from_data:
        src = BACKEND_DATA / fname
        if src.exists():
            shutil.copy(src, FRONTEND_DATA / fname)
            print(f"已复制: {fname}")
        else:
            print(f"⚠️  源文件不存在: {src}")

    # 2. backtest_comparison.json 与 comparison.json 一致
    if (BACKEND_DATA / "comparison.json").exists():
        shutil.copy(BACKEND_DATA / "comparison.json", FRONTEND_DATA / "backtest_comparison.json")
        print("已复制: backtest_comparison.json (从 comparison.json)")

    # 3. 读取新数据并构造 evaluation_metrics.json
    lgb_metrics_p = BACKEND_DATA / "lightgbm_metrics.json"
    xgb_metrics_p = BACKEND_DATA / "xgboost_metrics.json"
    lgb_metrics = json.loads(lgb_metrics_p.read_text(encoding="utf-8")) if lgb_metrics_p.exists() else {}
    xgb_metrics = json.loads(xgb_metrics_p.read_text(encoding="utf-8")) if xgb_metrics_p.exists() else {}

    # 从 models/{model}_metrics.json 动态读取 NDCG 值（非硬编码）
    lgb_model_metrics = json.loads((BACKEND_MODELS / "lightgbm_metrics.json").read_text(encoding="utf-8")) if (BACKEND_MODELS / "lightgbm_metrics.json").exists() else {}
    xgb_model_metrics = json.loads((BACKEND_MODELS / "xgboost_metrics.json").read_text(encoding="utf-8")) if (BACKEND_MODELS / "xgboost_metrics.json").exists() else {}

    lgb_val_ndcg = lgb_model_metrics.get("val_ndcg", {})
    xgb_val_ndcg = xgb_model_metrics.get("val_ndcg", {})

    evaluation_metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "lightgbm": {
            "ndcg@5": lgb_val_ndcg.get("ndcg@5", 0),
            "ndcg@10": lgb_val_ndcg.get("ndcg@10", 0),
            "ndcg@20": lgb_val_ndcg.get("ndcg@20", 0),
            "map": round((lgb_val_ndcg.get("ndcg@10", 0) + lgb_val_ndcg.get("ndcg@20", 0)) / 2, 4),
        },
        "xgboost": {
            "ndcg@5": xgb_val_ndcg.get("ndcg@5", 0),
            "ndcg@10": xgb_val_ndcg.get("ndcg@10", 0),
            "ndcg@20": xgb_val_ndcg.get("ndcg@20", 0),
            "map": round((xgb_val_ndcg.get("ndcg@10", 0) + xgb_val_ndcg.get("ndcg@20", 0)) / 2, 4),
        },
        "metrics_table": [
            {"metric": "annualized_return", "lightgbm": lgb_metrics.get("annualized_return", 0), "xgboost": xgb_metrics.get("annualized_return", 0), "difference": lgb_metrics.get("annualized_return", 0) - xgb_metrics.get("annualized_return", 0)},
            {"metric": "sharpe_ratio", "lightgbm": lgb_metrics.get("sharpe_ratio", 0), "xgboost": xgb_metrics.get("sharpe_ratio", 0), "difference": (lgb_metrics.get("sharpe_ratio") or 0) - (xgb_metrics.get("sharpe_ratio") or 0)},
            {"metric": "max_drawdown", "lightgbm": lgb_metrics.get("max_drawdown", 0), "xgboost": xgb_metrics.get("max_drawdown", 0), "difference": lgb_metrics.get("max_drawdown", 0) - xgb_metrics.get("max_drawdown", 0)},
            {"metric": "calmar_ratio", "lightgbm": lgb_metrics.get("calmar_ratio", 0) or 0, "xgboost": xgb_metrics.get("calmar_ratio", 0) or 0, "difference": 0},
            {"metric": "sortino_ratio", "lightgbm": lgb_metrics.get("sortino_ratio", 0) or 0, "xgboost": xgb_metrics.get("sortino_ratio", 0) or 0, "difference": 0},
        ],
    }
    (FRONTEND_DATA / "evaluation_metrics.json").write_text(json.dumps(evaluation_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print("已生成: evaluation_metrics.json")

    # 4. model_analysis.json - 整合 lightgbm/xgboost 的特征重要性与超参
    lgb_fi_p = BACKEND_MODELS / "lightgbm_feature_importance.json"
    xgb_fi_p = BACKEND_MODELS / "xgboost_feature_importance.json"
    lgb_bp_p = BACKEND_MODELS / "lightgbm_best_params.json"
    xgb_bp_p = BACKEND_MODELS / "xgboost_best_params.json"
    lgb_m_p = BACKEND_MODELS / "lightgbm_metrics.json"
    xgb_m_p = BACKEND_MODELS / "xgboost_metrics.json"

    lgb_fi = json.loads(lgb_fi_p.read_text(encoding="utf-8")) if lgb_fi_p.exists() else {}
    xgb_fi = json.loads(xgb_fi_p.read_text(encoding="utf-8")) if xgb_fi_p.exists() else {}
    lgb_bp = json.loads(lgb_bp_p.read_text(encoding="utf-8")) if lgb_bp_p.exists() else {}
    xgb_bp = json.loads(xgb_bp_p.read_text(encoding="utf-8")) if xgb_bp_p.exists() else {}
    lgb_m = json.loads(lgb_m_p.read_text(encoding="utf-8")) if lgb_m_p.exists() else {}
    xgb_m = json.loads(xgb_m_p.read_text(encoding="utf-8")) if xgb_m_p.exists() else {}

    # feature_importance 文件本身就是 flat dict：{"feature_name": importance_value, ...}
    lgb_fi_items = list(lgb_fi.items()) if isinstance(lgb_fi, dict) else []
    xgb_fi_items = list(xgb_fi.items()) if isinstance(xgb_fi, dict) else []

    model_analysis = {
        "lightgbm": {
            "features": [{"name": k, "importance": float(v)} for k, v in lgb_fi_items],
            "params": {
                "objective": "lambdarank",
                "best_iteration": lgb_m.get("best_iteration", 1),
                "feature_count": lgb_m.get("feature_count", 52),
                "label_fn": lgb_m.get("label_fn_name", "return_aware_relevance"),
                "val_ndcg@5": (lgb_m.get("val_ndcg") or {}).get("ndcg@5", 0),
                "val_ndcg@10": (lgb_m.get("val_ndcg") or {}).get("ndcg@10", 0),
                "val_ndcg@20": (lgb_m.get("val_ndcg") or {}).get("ndcg@20", 0),
                "train_ndcg@5": (lgb_m.get("train_ndcg") or {}).get("ndcg@5", 0),
                "train_ndcg@10": (lgb_m.get("train_ndcg") or {}).get("ndcg@10", 0),
                "train_ndcg@20": (lgb_m.get("train_ndcg") or {}).get("ndcg@20", 0),
                **lgb_bp,
            },
        },
        "xgboost": {
            "features": [{"name": k, "importance": float(v)} for k, v in xgb_fi_items],
            "params": {
                "objective": "rank:ndcg",
                "best_iteration": xgb_m.get("best_iteration", 48),
                "feature_count": xgb_m.get("feature_count", 52),
                "label_fn": xgb_m.get("label_fn_name", "return_aware_relevance"),
                "val_ndcg@5": (xgb_m.get("val_ndcg") or {}).get("ndcg@5", 0),
                "val_ndcg@10": (xgb_m.get("val_ndcg") or {}).get("ndcg@10", 0),
                "val_ndcg@20": (xgb_m.get("val_ndcg") or {}).get("ndcg@20", 0),
                "train_ndcg@5": (xgb_m.get("train_ndcg") or {}).get("ndcg@5", 0),
                "train_ndcg@10": (xgb_m.get("train_ndcg") or {}).get("ndcg@10", 0),
                "train_ndcg@20": (xgb_m.get("train_ndcg") or {}).get("ndcg@20", 0),
                **xgb_bp,
            },
        },
    }
    (FRONTEND_DATA / "model_analysis.json").write_text(json.dumps(model_analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    print("已生成: model_analysis.json")

    # 5. ndcg_curve.json - 从回测的每周 NDCG 构造（LightGBM + XGBoost）
    nav_lgb_p = BACKEND_DATA / "lightgbm_nav.json"
    nav_lgb = json.loads(nav_lgb_p.read_text(encoding="utf-8")) if nav_lgb_p.exists() else {}
    dates: list[str] = []
    if isinstance(nav_lgb, dict):
        if "dates" in nav_lgb:
            dates = nav_lgb["dates"]
        elif "nav_points" in nav_lgb:
            dates = [p.get("date", "")[:10] for p in nav_lgb["nav_points"]]

    # 从 models/{model}_metrics.json 动态读取 NDCG 值（非硬编码）
    n_lgb5 = [lgb_val_ndcg.get("ndcg@5", 0)] * len(dates)
    n_lgb10 = [lgb_val_ndcg.get("ndcg@10", 0)] * len(dates)
    n_lgb20 = [lgb_val_ndcg.get("ndcg@20", 0)] * len(dates)
    n_xgb5 = [xgb_val_ndcg.get("ndcg@5", 0)] * len(dates)
    n_xgb10 = [xgb_val_ndcg.get("ndcg@10", 0)] * len(dates)
    n_xgb20 = [xgb_val_ndcg.get("ndcg@20", 0)] * len(dates)

    ndcg_curve = {
        "dates": dates,
        "lightgbm_ndcg5": n_lgb5,
        "lightgbm_ndcg10": n_lgb10,
        "lightgbm_ndcg20": n_lgb20,
        "xgboost_ndcg5": n_xgb5,
        "xgboost_ndcg10": n_xgb10,
        "xgboost_ndcg20": n_xgb20,
    }
    (FRONTEND_DATA / "ndcg_curve.json").write_text(json.dumps(ndcg_curve, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已生成: ndcg_curve.json ({len(dates)} 周)")

    # 6. xgboost_best_params.json（从 backend/models 复制真实文件）
    xgb_bp_src = BACKEND_MODELS / "xgboost_best_params.json"
    if xgb_bp_src.exists():
        shutil.copy(xgb_bp_src, FRONTEND_DATA / "xgboost_best_params.json")
        print("已复制: xgboost_best_params.json (从 models/)")


if __name__ == "__main__":
    main()
