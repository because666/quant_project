# [共享文件] 本文件同时存在于 project/backend/src/ 和 thesis_experiments/src/，修改时请同步更新两处
"""
XGBoost rank:ndcg 训练与 Optuna 超参搜索（与 LightGBM 共用 data_loader 与 relevance 映射）。

用法::

    python -m src.model_xgboost --trials 20
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from xgboost.core import XGBoostError

from .data_loader import DATA_OUT_DIR, load_training_data, load_validation_data
from .model_lightgbm import future_return_to_relevance, return_aware_relevance


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "xgboost.json"
DEFAULT_IMPORTANCE_PATH = MODELS_DIR / "xgboost_feature_importance.json"
DEFAULT_LOG_PATH = MODELS_DIR / "xgboost_training.log"
DEFAULT_TUNE_LOG_PATH = MODELS_DIR / "xgboost_optuna_trials.jsonl"

RANDOM_STATE = 42
N_OPTUNA_TRIALS = 20
EARLY_STOPPING_ROUNDS = 100
MAX_BOOST_ROUND = 2000
_EXCLUDE_COLS = {"group_id", "group_size"}


def _setup_file_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("quant_xgboost")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def get_base_params() -> dict[str, Any]:
    """
    训练基础参数（固定随机种子 42，可复现）。

    默认采用较强的正则化（max_depth=4、min_child_weight=20、gamma=0.1、
    alpha=1.0、reg_lambda=2.0、subsample=0.6、colsample_bytree=0.6），
    配合 E1a 收益加权标签使用，避免 52 因子 + 62 周数据下的过拟合。
    """
    return {
        "objective": "rank:ndcg",
        "eval_metric": ["ndcg@5", "ndcg@20", "ndcg@10"],
        "ndcg_exp_gain": False,
        "booster": "gbtree",
        "eta": 0.05,
        "max_depth": 4,
        "subsample": 0.6,
        "colsample_bytree": 0.6,
        "min_child_weight": 20,
        "gamma": 0.1,
        "alpha": 1.0,
        "lambda": 2.0,
        "seed": RANDOM_STATE,
    }


def build_dmats(
    *,
    data_dir: Path = DATA_OUT_DIR,
    fill_missing: bool = True,
) -> tuple[xgb.DMatrix, xgb.DMatrix, list[str], np.ndarray, np.ndarray]:
    """与 LightGBM 相同数据源；group 为各 query 样本数列表。"""
    X_tr, y_tr, g_tr = load_training_data(data_dir=data_dir, fill_missing=fill_missing)
    X_va, y_va, g_va = load_validation_data(data_dir=data_dir, fill_missing=fill_missing)

    if sum(g_tr) != len(X_tr) or sum(g_va) != len(X_va):
        raise ValueError(
            f"group 与样本数不一致: train {sum(g_tr)} vs {len(X_tr)}, val {sum(g_va)} vs {len(X_va)}"
        )

    feat_names = [c for c in X_tr.columns if c not in _EXCLUDE_COLS]
    X_tr = X_tr[feat_names]
    X_va = X_va[feat_names]
    X_tr_m = np.ascontiguousarray(X_tr.to_numpy(dtype=np.float32, copy=True))
    X_va_m = np.ascontiguousarray(X_va.to_numpy(dtype=np.float32, copy=True))

    y_tr_rel = future_return_to_relevance(y_tr, g_tr)
    y_va_rel = future_return_to_relevance(y_va, g_va)

    group_tr = np.asarray(g_tr, dtype=np.uint32)
    group_va = np.asarray(g_va, dtype=np.uint32)

    dtrain = xgb.DMatrix(X_tr_m, label=y_tr_rel, feature_names=feat_names)
    dtrain.set_group(group_tr)
    dval = xgb.DMatrix(X_va_m, label=y_va_rel, feature_names=feat_names)
    dval.set_group(group_va)
    return dtrain, dval, feat_names, y_tr, y_va


def build_dmats_with_label_fn(
    label_fn: Callable[..., np.ndarray],
    *,
    data_dir: Path = DATA_OUT_DIR,
    fill_missing: bool = True,
    label_fn_kwargs: dict[str, Any] | None = None,
) -> tuple[xgb.DMatrix, xgb.DMatrix, list[str], np.ndarray, np.ndarray]:
    """
    使用自定义标签构造函数构建 xgb.DMatrix（训练 / 验证）。

    与 build_dmats 逻辑相同，但支持传入自定义标签函数替代
    future_return_to_relevance。若标签函数需要额外参数（如 volatility），
    可通过 label_fn_kwargs 传入。

    对于 E1b/E1c 标签函数需要 volatility 数据时，会自动从因子列中
    提取 volatility_12w 并传入 label_fn_kwargs。

    参数:
        label_fn: 标签构造函数，签名需兼容 (y, group_sizes, **kwargs) -> np.ndarray
        data_dir: 数据目录
        fill_missing: 是否填充缺失值
        label_fn_kwargs: 传递给 label_fn 的额外关键字参数

    返回:
        (dtrain, dval, feature_names, y_train_raw, y_val_raw)
    """
    X_tr, y_tr, g_tr = load_training_data(data_dir=data_dir, fill_missing=fill_missing)
    X_va, y_va, g_va = load_validation_data(data_dir=data_dir, fill_missing=fill_missing)

    if sum(g_tr) != len(X_tr) or sum(g_va) != len(X_va):
        raise ValueError(
            f"group 与样本数不一致: train {sum(g_tr)} vs {len(X_tr)}, val {sum(g_va)} vs {len(X_va)}"
        )

    feat_names = [c for c in X_tr.columns if c not in _EXCLUDE_COLS]

    kw: dict[str, Any] = dict(label_fn_kwargs) if label_fn_kwargs else {}
    if "volatility" not in kw and "volatility_12w" in X_tr.columns:
        vol_tr = pd.to_numeric(X_tr["volatility_12w"], errors="coerce").to_numpy(dtype=np.float64)
        vol_va = pd.to_numeric(X_va["volatility_12w"], errors="coerce").to_numpy(dtype=np.float64)
        vol_tr = np.nan_to_num(vol_tr, nan=0.0)
        vol_va = np.nan_to_num(vol_va, nan=0.0)
        kw["volatility"] = vol_tr
        kw_va: dict[str, Any] = dict(kw)
        kw_va["volatility"] = vol_va
    else:
        kw_va = dict(kw)

    y_tr_rel = label_fn(y_tr, g_tr, **kw)
    y_va_rel = label_fn(y_va, g_va, **kw_va)

    X_tr = X_tr[feat_names]
    X_va = X_va[feat_names]
    X_tr_m = np.ascontiguousarray(X_tr.to_numpy(dtype=np.float32, copy=True))
    X_va_m = np.ascontiguousarray(X_va.to_numpy(dtype=np.float32, copy=True))

    group_tr = np.asarray(g_tr, dtype=np.uint32)
    group_va = np.asarray(g_va, dtype=np.uint32)

    dtrain = xgb.DMatrix(X_tr_m, label=y_tr_rel, feature_names=feat_names)
    dtrain.set_group(group_tr)
    dval = xgb.DMatrix(X_va_m, label=y_va_rel, feature_names=feat_names)
    dval.set_group(group_va)
    return dtrain, dval, feat_names, y_tr, y_va


def _ndcg_at_best_iteration(evals_result: dict[str, Any], split: str, best_iteration: int) -> dict[str, float]:
    sp = evals_result[split]
    i = int(best_iteration)
    i = max(0, min(i, len(sp["ndcg@10"]) - 1))
    return {
        "ndcg@5": float(sp["ndcg@5"][i]),
        "ndcg@10": float(sp["ndcg@10"][i]),
        "ndcg@20": float(sp["ndcg@20"][i]),
    }


def train_booster(
    params: dict[str, Any],
    dtrain: xgb.DMatrix,
    dval: xgb.DMatrix,
    *,
    num_boost_round: int = MAX_BOOST_ROUND,
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
    verbose_eval: bool | int = False,
) -> tuple[xgb.Booster, dict[str, Any]]:
    """
    evals 顺序：train 在前、val 在后，使早停依据最后一项验证集（XGBoost 约定）。
    """
    evals_result: dict[str, Any] = {}
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=verbose_eval,
        evals_result=evals_result,
    )
    return bst, evals_result


@dataclass
class TuneResult:
    best_params: dict[str, Any]
    best_val_ndcg10: float
    n_trials: int


def tune_xgboost(
    dtrain: xgb.DMatrix,
    dval: xgb.DMatrix,
    *,
    base_params: dict[str, Any] | None = None,
    n_trials: int = N_OPTUNA_TRIALS,
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
    seed: int = RANDOM_STATE,
    trials_log_path: Path | None = DEFAULT_TUNE_LOG_PATH,
    logger: logging.Logger | None = None,
) -> TuneResult:
    """
    Optuna 最大化验证集 NDCG@10；每 trial 使用 early_stopping_rounds。

    强正则化搜索空间（针对 52 因子 62 周数据严重过拟合问题）：
    - max_depth: 3-8（限制单树深度）
    - eta: 0.02-0.08（log 尺度，控制学习率）
    - min_child_weight: 10-200（避免过拟合小样本）
    - gamma: 0-5（最小分裂损失）
    - alpha (L1): 0.5-10.0
    - reg_lambda (L2): 0.5-10.0
    - subsample: 0.4-0.8（样本子采样）
    - colsample_bytree: 0.4-0.8（特征子采样）
    """
    base = dict(get_base_params() if base_params is None else base_params)
    # XGBoost 中 L2 正则参数为 "lambda"，避免与 Python 关键字冲突，
    # 在 Optuna 中建议名称使用 reg_lambda（与 LightGBM 风格一致），
    # 内部显式映射到 params["lambda"]
    base_lambda = base.get("lambda", 2.0)
    if "lambda" in base:
        base_clean = {k: v for k, v in base.items() if k != "lambda"}
    else:
        base_clean = dict(base)

    def objective(trial: optuna.Trial) -> float:
        params = {
            **base_clean,
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "eta": trial.suggest_float("eta", 0.02, 0.08, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 10, 200),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "alpha": trial.suggest_float("alpha", 0.5, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 10.0),
            "subsample": trial.suggest_float("subsample", 0.4, 0.8),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.8),
            "lambda": trial.suggest_float("reg_lambda", 0.5, 10.0),
        }
        bst, er = train_booster(
            params,
            dtrain,
            dval,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,
        )
        bi = int(bst.best_iteration)
        score = float(er["val"]["ndcg@10"][bi])
        trial.set_user_attr("best_iteration", bi)
        trial.set_user_attr("val_ndcg@5", float(er["val"]["ndcg@5"][bi]))
        trial.set_user_attr("val_ndcg@20", float(er["val"]["ndcg@20"][bi]))
        if trials_log_path is not None:
            trials_log_path.parent.mkdir(parents=True, exist_ok=True)
            rec = {
                "trial": trial.number,
                "value": score,
                "params": {
                    "max_depth": params["max_depth"],
                    "eta": params["eta"],
                    "min_child_weight": params["min_child_weight"],
                    "gamma": params["gamma"],
                    "alpha": params["alpha"],
                    "reg_lambda": params["reg_lambda"],
                    "subsample": params["subsample"],
                    "colsample_bytree": params["colsample_bytree"],
                },
                "user_attrs": dict(trial.user_attrs),
            }
            with open(trials_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return score

    if trials_log_path is not None and trials_log_path.exists():
        trials_log_path.unlink()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_trial
    best_reg_lambda = float(best.params["reg_lambda"])
    best_params = {
        **base_clean,
        "max_depth": int(best.params["max_depth"]),
        "eta": float(best.params["eta"]),
        "min_child_weight": int(best.params["min_child_weight"]),
        "gamma": float(best.params["gamma"]),
        "alpha": float(best.params["alpha"]),
        "reg_lambda": best_reg_lambda,
        "lambda": best_reg_lambda,
        "subsample": float(best.params["subsample"]),
        "colsample_bytree": float(best.params["colsample_bytree"]),
    }
    msg = f"Optuna 完成: n_trials={n_trials}, best_val_ndcg@10={best.value:.6f}, best_params={best.params}"
    if logger:
        logger.info(msg)
    else:
        print(msg)

    return TuneResult(best_params=best_params, best_val_ndcg10=float(best.value), n_trials=n_trials)


def save_feature_importance_json(bst: xgb.Booster, feature_names: list[str], path: Path) -> None:
    raw = bst.get_score(importance_type="gain")
    payload = {str(n): float(raw.get(str(n), 0.0)) for n in feature_names}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _save_xgb_model(bst: xgb.Booster, model_path: Path) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        str(model_path.resolve()).encode("ascii")
        bst.save_model(str(model_path.resolve()))
    except (UnicodeEncodeError, OSError, XGBoostError):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, dir=str(model_path.parent)) as tmp:
            tmp_name = tmp.name
        bst.save_model(tmp_name)
        Path(tmp_name).replace(model_path)


def train_final_xgboost_with_label_fn(
    label_fn: Callable[..., np.ndarray],
    *,
    label_fn_name: str = "custom",
    label_fn_kwargs: dict[str, Any] | None = None,
    data_dir: Path = DATA_OUT_DIR,
    model_path: Path | None = None,
    importance_path: Path | None = None,
    log_path: Path | None = None,
    tune: bool = True,
    n_trials: int = N_OPTUNA_TRIALS,
    fill_missing: bool = True,
    tune_seed: int = RANDOM_STATE,
) -> tuple[xgb.Booster, dict[str, Any]]:
    """
    使用自定义标签构造函数训练 XGBoost rank:ndcg 模型。

    与 train_final_xgboost 逻辑相同，但支持传入自定义标签函数。
    模型默认保存为 models/{label_fn_name}_xgboost.json，
    指标保存为 models/{label_fn_name}_xgboost_metrics.json。

    参数:
        label_fn: 标签构造函数，签名需兼容 (y, group_sizes, **kwargs) -> np.ndarray
        label_fn_name: 标签函数名称，用于生成默认保存路径
        label_fn_kwargs: 传递给 label_fn 的额外关键字参数
        data_dir: 数据目录
        model_path: 模型保存路径，为 None 时使用 models/{label_fn_name}_xgboost.json
        importance_path: 特征重要性保存路径
        log_path: 训练日志路径
        tune: 是否进行 Optuna 超参搜索
        n_trials: Optuna 搜索次数
        fill_missing: 是否填充缺失值
        tune_seed: Optuna TPESampler 种子

    返回:
        (训练好的 Booster, 最终使用的参数字典)
    """
    if model_path is None:
        model_path = MODELS_DIR / f"{label_fn_name}_xgboost.json"
    if importance_path is None:
        importance_path = MODELS_DIR / f"{label_fn_name}_xgboost_feature_importance.json"
    if log_path is None:
        log_path = MODELS_DIR / f"{label_fn_name}_xgboost_training.log"

    logger = _setup_file_logger(log_path)
    logger.info(
        "开始 XGBoost rank:ndcg（标签函数=%s）：加载数据（fill_missing=%s）",
        label_fn_name, fill_missing,
    )
    dtrain, dval, feat_names, _, _ = build_dmats_with_label_fn(
        label_fn,
        data_dir=data_dir,
        fill_missing=fill_missing,
        label_fn_kwargs=label_fn_kwargs,
    )

    base = get_base_params()
    if tune:
        logger.info(
            "超参数搜索：%d 次 trial，标签函数=%s，目标=验证集 NDCG@10，early_stopping=%d",
            n_trials, label_fn_name, EARLY_STOPPING_ROUNDS,
        )
        tune_res = tune_xgboost(
            dtrain,
            dval,
            base_params=base,
            n_trials=n_trials,
            seed=tune_seed,
            logger=logger,
        )
        final_params = tune_res.best_params
        logger.info("最优超参: %s", final_params)
        with open(MODELS_DIR / f"{label_fn_name}_xgboost_best_params.json", "w", encoding="utf-8") as f:
            json.dump(
                {k: v for k, v in final_params.items() if isinstance(v, (int, float, str, bool, list))},
                f, indent=2,
            )
    else:
        final_params = base
        logger.info("跳过调优，使用强正则化基础参数")

    logger.info("使用最终参数训练并早停（early_stopping_rounds=%d）", EARLY_STOPPING_ROUNDS)
    bst, er = train_booster(
        final_params,
        dtrain,
        dval,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=50,
    )
    bi = int(bst.best_iteration)
    val_m = _ndcg_at_best_iteration(er, "val", bi)
    tr_m = _ndcg_at_best_iteration(er, "train", bi)
    logger.info(
        "验证集 NDCG: @5=%.6f @10=%.6f @20=%.6f | 训练集 NDCG: @5=%.6f @10=%.6f @20=%.6f | best_iteration=%d",
        val_m["ndcg@5"],
        val_m["ndcg@10"],
        val_m["ndcg@20"],
        tr_m["ndcg@5"],
        tr_m["ndcg@10"],
        tr_m["ndcg@20"],
        bi,
    )

    _save_xgb_model(bst, model_path)
    logger.info("模型已保存: %s", model_path)

    save_feature_importance_json(bst, feat_names, importance_path)
    logger.info("特征重要性已保存: %s", importance_path)

    meta = {
        "label_fn_name": label_fn_name,
        "val_ndcg": val_m,
        "train_ndcg": tr_m,
        "best_iteration": bi,
        "feature_count": len(feat_names),
        "features": feat_names,
        "train_val_gap": {
            "ndcg@5": float(tr_m["ndcg@5"] - val_m["ndcg@5"]),
            "ndcg@10": float(tr_m["ndcg@10"] - val_m["ndcg@10"]),
            "ndcg@20": float(tr_m["ndcg@20"] - val_m["ndcg@20"]),
        },
    }
    metrics_path = MODELS_DIR / f"{label_fn_name}_xgboost_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    logger.info("指标已保存: %s", metrics_path)

    return bst, final_params


def train_final_xgboost_e1a(
    *,
    data_dir: Path = DATA_OUT_DIR,
    model_path: Path | None = None,
    importance_path: Path | None = None,
    log_path: Path | None = None,
    tune: bool = True,
    n_trials: int = N_OPTUNA_TRIALS,
    fill_missing: bool = True,
    tune_seed: int = RANDOM_STATE,
) -> tuple[xgb.Booster, dict[str, Any]]:
    """
    使用 E1a 收益加权标签训练 XGBoost rank:ndcg 模型。

    该函数是对 train_final_xgboost_with_label_fn 的便捷封装，
    内部固定使用 return_aware_relevance 作为标签构造函数，
    配合 get_base_params 的强正则化默认参数，缓解过拟合。

    模型默认保存为 models/xgboost.json（与 train_final_xgboost 一致），
    指标保存为 models/xgboost_metrics.json（额外包含 label_fn_name 字段）。

    参数:
        data_dir: 数据目录
        model_path: 模型保存路径，None 时使用 models/xgboost.json
        importance_path: 特征重要性保存路径
        log_path: 训练日志保存路径
        tune: 是否进行 Optuna 超参搜索
        n_trials: Optuna 搜索次数
        fill_missing: 是否填充缺失值
        tune_seed: Optuna TPESampler 种子

    返回:
        (训练好的 Booster, 最终使用的参数字典)
    """
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    if importance_path is None:
        importance_path = DEFAULT_IMPORTANCE_PATH
    if log_path is None:
        log_path = DEFAULT_LOG_PATH

    logger = _setup_file_logger(log_path)
    logger.info("开始 XGBoost rank:ndcg（E1a 收益加权标签）：加载数据（fill_missing=%s）", fill_missing)
    dtrain, dval, feat_names, _, _ = build_dmats_with_label_fn(
        return_aware_relevance,
        data_dir=data_dir,
        fill_missing=fill_missing,
    )

    base = get_base_params()
    if tune:
        logger.info(
            "超参数搜索（强正则化空间）：%d 次 trial，目标=验证集 NDCG@10，early_stopping=%d",
            n_trials, EARLY_STOPPING_ROUNDS,
        )
        tune_res = tune_xgboost(
            dtrain,
            dval,
            base_params=base,
            n_trials=n_trials,
            seed=tune_seed,
            logger=logger,
        )
        final_params = tune_res.best_params
        logger.info("最优超参: %s", final_params)
        with open(MODELS_DIR / "xgboost_best_params.json", "w", encoding="utf-8") as f:
            json.dump(
                {k: v for k, v in final_params.items() if isinstance(v, (int, float, str, bool, list))},
                f, indent=2,
            )
    else:
        final_params = base
        logger.info("跳过调优，使用强正则化基础参数")

    logger.info("使用最终参数训练并早停（early_stopping_rounds=%d）", EARLY_STOPPING_ROUNDS)
    bst, er = train_booster(
        final_params,
        dtrain,
        dval,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=50,
    )
    bi = int(bst.best_iteration)
    val_m = _ndcg_at_best_iteration(er, "val", bi)
    tr_m = _ndcg_at_best_iteration(er, "train", bi)
    logger.info(
        "验证集 NDCG: @5=%.6f @10=%.6f @20=%.6f | 训练集 NDCG: @5=%.6f @10=%.6f @20=%.6f | best_iteration=%d",
        val_m["ndcg@5"],
        val_m["ndcg@10"],
        val_m["ndcg@20"],
        tr_m["ndcg@5"],
        tr_m["ndcg@10"],
        tr_m["ndcg@20"],
        bi,
    )

    _save_xgb_model(bst, model_path)
    logger.info("模型已保存: %s", model_path)

    save_feature_importance_json(bst, feat_names, importance_path)
    logger.info("特征重要性已保存: %s", importance_path)

    meta = {
        "label_fn_name": "return_aware_relevance",
        "val_ndcg": val_m,
        "train_ndcg": tr_m,
        "best_iteration": bi,
        "feature_count": len(feat_names),
        "features": feat_names,
        "train_val_gap": {
            "ndcg@5": float(tr_m["ndcg@5"] - val_m["ndcg@5"]),
            "ndcg@10": float(tr_m["ndcg@10"] - val_m["ndcg@10"]),
            "ndcg@20": float(tr_m["ndcg@20"] - val_m["ndcg@20"]),
        },
    }
    metrics_path = MODELS_DIR / "xgboost_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    logger.info("指标已保存: %s", metrics_path)

    return bst, final_params


def train_final_xgboost(
    *,
    data_dir: Path = DATA_OUT_DIR,
    model_path: Path = DEFAULT_MODEL_PATH,
    importance_path: Path = DEFAULT_IMPORTANCE_PATH,
    log_path: Path = DEFAULT_LOG_PATH,
    tune: bool = True,
    n_trials: int = N_OPTUNA_TRIALS,
    fill_missing: bool = True,
) -> tuple[xgb.Booster, dict[str, Any]]:
    """
    训练 XGBoost rank:ndcg 模型（使用 future_return_to_relevance 纯排名标签）。

    该函数保留作为旧版入口；新代码应优先使用 train_final_xgboost_e1a。

    参数:
        data_dir: 数据目录
        model_path: 模型保存路径
        importance_path: 特征重要性保存路径
        log_path: 训练日志保存路径
        tune: 是否进行 Optuna 超参搜索
        n_trials: Optuna 搜索次数
        fill_missing: 是否填充缺失值

    返回:
        (训练好的 Booster, 最终使用的参数字典)
    """
    logger = _setup_file_logger(log_path)
    logger.info("开始 XGBoost rank:ndcg：加载数据（fill_missing=%s）", fill_missing)
    dtrain, dval, feat_names, _, _ = build_dmats(data_dir=data_dir, fill_missing=fill_missing)

    base = get_base_params()
    if tune:
        logger.info("超参数搜索：%d 次 trial，目标=验证集 NDCG@10，early_stopping=%d", n_trials, EARLY_STOPPING_ROUNDS)
        tune_res = tune_xgboost(
            dtrain,
            dval,
            base_params=base,
            n_trials=n_trials,
            logger=logger,
        )
        final_params = tune_res.best_params
        logger.info("最优超参: %s", final_params)
        with open(MODELS_DIR / "xgboost_best_params.json", "w", encoding="utf-8") as f:
            json.dump(
                {k: v for k, v in final_params.items() if isinstance(v, (int, float, str, bool, list))},
                f,
                indent=2,
            )
    else:
        final_params = base
        logger.info("跳过调优，使用基础参数")

    logger.info("使用最终参数训练并早停（early_stopping_rounds=%d）", EARLY_STOPPING_ROUNDS)
    bst, er = train_booster(
        final_params,
        dtrain,
        dval,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=50,
    )
    bi = int(bst.best_iteration)
    val_m = _ndcg_at_best_iteration(er, "val", bi)
    tr_m = _ndcg_at_best_iteration(er, "train", bi)
    logger.info(
        "验证集 NDCG: @5=%.6f @10=%.6f @20=%.6f | 训练集 NDCG: @5=%.6f @10=%.6f @20=%.6f | best_iteration=%d",
        val_m["ndcg@5"],
        val_m["ndcg@10"],
        val_m["ndcg@20"],
        tr_m["ndcg@5"],
        tr_m["ndcg@10"],
        tr_m["ndcg@20"],
        bi,
    )

    _save_xgb_model(bst, model_path)
    logger.info("模型已保存: %s", model_path)

    save_feature_importance_json(bst, feat_names, importance_path)
    logger.info("特征重要性已保存: %s", importance_path)

    meta = {
        "label_fn_name": "future_return_to_relevance",
        "val_ndcg": val_m,
        "train_ndcg": tr_m,
        "best_iteration": bi,
        "feature_count": len(feat_names),
        "features": feat_names,
    }
    with open(MODELS_DIR / "xgboost_metrics.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return bst, final_params


def load_xgboost_model(path: Path | None = None) -> xgb.Booster:
    p = path or DEFAULT_MODEL_PATH
    if not p.exists():
        raise FileNotFoundError(str(p))
    try:
        return xgb.Booster(model_file=str(p.resolve()))
    except XGBoostError:
        return xgb.Booster(model_file=p.read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost rank:ndcg with optional Optuna tuning.")
    parser.add_argument("--trials", type=int, default=N_OPTUNA_TRIALS, help="Optuna trials (default 20)")
    parser.add_argument("--no-tune", action="store_true", help="Skip hyperparameter search")
    parser.add_argument("--data-dir", type=str, default=None, help="数据目录（默认data/）")
    parser.add_argument(
        "--label-fn",
        type=str,
        default="return_aware_relevance",
        choices=["return_aware_relevance", "future_return_to_relevance"],
        help="标签函数：return_aware_relevance（E1a 收益加权，默认）或 future_return_to_relevance（纯排名）",
    )
    args = parser.parse_args()
    if args.label_fn == "return_aware_relevance":
        train_final_xgboost_e1a(
            tune=not args.no_tune,
            n_trials=max(1, args.trials),
            data_dir=Path(args.data_dir) if args.data_dir else DATA_OUT_DIR,
        )
    else:
        train_final_xgboost(
            tune=not args.no_tune,
            n_trials=max(1, args.trials),
            data_dir=Path(args.data_dir) if args.data_dir else DATA_OUT_DIR,
        )


if __name__ == "__main__":
    main()
