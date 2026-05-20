# [共享文件] 本文件同时存在于 project/backend/src/ 和 thesis_experiments/src/，修改时请同步更新两处
"""
LightGBM LambdaRank 训练与 Optuna 超参搜索。

用法（在 backend 目录且 PYTHONPATH 含当前项目根）::

    python -m src.model_lightgbm --trials 20

说明：lambdarank 需离散 relevance 标签，本模块将截面 future_return 映射为 0–30；
验证集 NDCG 高度依赖标签与因子质量，未必总能超过 0.5。
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import lightgbm as lgb
from lightgbm.basic import LightGBMError
import numpy as np
import optuna
import pandas as pd
from .data_loader import DATA_OUT_DIR, load_training_data, load_validation_data


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "lightgbm.pkl"
DEFAULT_IMPORTANCE_PATH = MODELS_DIR / "lightgbm_feature_importance.json"
DEFAULT_LOG_PATH = MODELS_DIR / "lightgbm_training.log"
DEFAULT_TUNE_LOG_PATH = MODELS_DIR / "lightgbm_optuna_trials.jsonl"

RANDOM_STATE = 42
N_OPTUNA_TRIALS = 20
EARLY_STOPPING_ROUNDS = 50
MAX_BOOST_ROUND = 2000


def _setup_file_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("quant_lightgbm")
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


def future_return_to_relevance(y: np.ndarray, group_sizes: list[int], max_label: int = 30) -> np.ndarray:
    """
    将连续 future_return 按 query 内分位映射为整数 relevance，满足 LightGBM lambdarank 标签要求（0..max_label）。
    同组内收益越高，relevance 越大。
    """
    y = np.asarray(y, dtype=np.float64)
    out = np.zeros(len(y), dtype=np.int32)
    pos = 0
    for gsz in group_sizes:
        if gsz <= 0:
            continue
        sl = slice(pos, pos + gsz)
        seg = y[sl]
        if gsz == 1:
            out[sl] = max_label // 2
        else:
            order = np.argsort(-seg, kind="mergesort")
            rank = np.empty(gsz, dtype=np.int64)
            rank[order] = np.arange(gsz)
            rel = (max_label * (1.0 - rank / (gsz - 1.0))).round().astype(np.int32)
            out[sl] = np.clip(rel, 0, max_label)
        pos += gsz
    return out


def return_aware_relevance(
    y: np.ndarray,
    group_sizes: list[int],
    max_label: int = 30,
    alpha: float = 1.0,
) -> np.ndarray:
    """
    E1a：收益加权 NDCG 标签。

    在标准排名标签基础上，根据实际收益率大小额外提升高收益股票的标签值，
    使排序学习更关注绝对收益水平而非仅相对排名。

    参数:
        y: 未来收益率数组，形状 (N,)
        group_sizes: 每个截面（query）的样本数列表
        max_label: 标签上界，默认 30
        alpha: 收益加权的幂次，控制奖励强度，默认 1.0

    返回:
        整数 relevance 标签数组，形状 (N,)，值域 [0, max_label]
    """
    y = np.asarray(y, dtype=np.float64)
    rank_rel = future_return_to_relevance(y, group_sizes, max_label=max_label).astype(np.float64)
    out = np.zeros(len(y), dtype=np.float64)
    pos = 0
    for gsz in group_sizes:
        if gsz <= 0:
            continue
        sl = slice(pos, pos + gsz)
        seg = y[sl]
        abs_seg = np.abs(seg)
        bonus = np.power(abs_seg, alpha)
        bonus_max = bonus.max()
        if bonus_max > 0:
            bonus = bonus / bonus_max * max_label * 0.3
        else:
            bonus = np.zeros(gsz, dtype=np.float64)
        out[sl] = rank_rel[sl] + bonus
        pos += gsz
    return np.clip(out, 0, max_label).astype(np.int32)


def sharpe_aware_relevance(
    y: np.ndarray,
    group_sizes: list[int],
    max_label: int = 30,
    alpha: float = 1.0,
    volatility: np.ndarray | None = None,
    sharpe_penalty: float = 0.5,
) -> np.ndarray:
    """
    E1b：收益加权 + 夏普惩罚标签。

    在 E1a 基础上，对高波动股票的标签进行惩罚，使模型倾向于选择
    收益/风险比更优的股票。

    参数:
        y: 未来收益率数组，形状 (N,)
        group_sizes: 每个截面（query）的样本数列表
        max_label: 标签上界，默认 30
        alpha: 收益加权的幂次，控制奖励强度，默认 1.0
        volatility: 波动率数组（如 volatility_12w），形状 (N,)，
                    为 None 时退化为 E1a
        sharpe_penalty: 夏普惩罚系数，默认 0.5（原0.1过小，整数截断后几乎无差异）

    返回:
        整数 relevance 标签数组，形状 (N,)，值域 [0, max_label]
    """
    e1a_labels = return_aware_relevance(y, group_sizes, max_label=max_label, alpha=alpha).astype(np.float64)
    if volatility is None:
        return e1a_labels.astype(np.int32)
    volatility = np.asarray(volatility, dtype=np.float64)
    if len(volatility) != len(y):
        raise ValueError(f"volatility 长度 {len(volatility)} 与 y 长度 {len(y)} 不一致")
    out = e1a_labels.copy()
    pos = 0
    for gsz in group_sizes:
        if gsz <= 0:
            continue
        sl = slice(pos, pos + gsz)
        vol_seg = volatility[sl]
        vol_max = vol_seg.max()
        if vol_max > 0:
            penalty = sharpe_penalty * (vol_seg / vol_max) * max_label * 0.5
            out[sl] = out[sl] - penalty
        pos += gsz
    return np.clip(out, 0, max_label).astype(np.int32)


def cvar_aware_relevance(
    y: np.ndarray,
    group_sizes: list[int],
    max_label: int = 30,
    alpha: float = 1.0,
    volatility: np.ndarray | None = None,
    sharpe_penalty: float = 0.5,
    cvar_penalty: float = 1.5,
    cvar_alpha: float = 0.20,
) -> np.ndarray:
    """
    E1c：收益加权 + 夏普惩罚 + CVaR 惩罚标签。

    在 E1b 基础上，对左尾收益（极端亏损）的股票额外惩罚，
    使模型更加规避下行风险。

    参数:
        y: 未来收益率数组，形状 (N,)
        group_sizes: 每个截面（query）的样本数列表
        max_label: 标签上界，默认 30
        alpha: 收益加权的幂次，默认 1.0
        volatility: 波动率数组（如 volatility_12w），形状 (N,)，
                    为 None 时退化为 E1a
        sharpe_penalty: 夏普惩罚系数，默认 0.5
        cvar_penalty: CVaR 惩罚系数，默认 1.5（0.3过弱导致E1c与E1b无差异）
        cvar_alpha: CVaR 分位数阈值，默认 0.20（即 20% 分位数，0.10覆盖太少）

    返回:
        整数 relevance 标签数组，形状 (N,)，值域 [0, max_label]
    """
    e1b_labels = sharpe_aware_relevance(
        y, group_sizes,
        max_label=max_label, alpha=alpha,
        volatility=volatility, sharpe_penalty=sharpe_penalty,
    ).astype(np.float64)
    y = np.asarray(y, dtype=np.float64)
    out = e1b_labels.copy()
    pos = 0
    for gsz in group_sizes:
        if gsz <= 0:
            continue
        sl = slice(pos, pos + gsz)
        seg = y[sl]
        threshold = np.quantile(seg, cvar_alpha)
        below_mask = seg < threshold
        if np.any(below_mask) and abs(threshold) > 1e-12:
            distance = np.maximum((threshold - seg) / abs(threshold), 0.0)
            cvar_pen = cvar_penalty * distance * max_label
            out[sl] = out[sl] - cvar_pen
        pos += gsz
    return np.clip(out, 0, max_label).astype(np.int32)


def get_base_params() -> dict[str, Any]:
    """训练基础参数（固定随机种子 42，可复现）。"""
    return {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [5, 10, 20],
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 50,
        "lambda_l1": 0.5,
        "lambda_l2": 0.5,
        "verbose": -1,
        "seed": RANDOM_STATE,
        "bagging_seed": RANDOM_STATE,
        "feature_fraction_seed": RANDOM_STATE,
        "deterministic": True,
        # Optuna 会改变 min_child_samples，需关闭预过滤以免 C API 报错
        "feature_pre_filter": False,
    }


def build_datasets(
    *,
    data_dir: Path = DATA_OUT_DIR,
    fill_missing: bool = True,
) -> tuple[lgb.Dataset, lgb.Dataset, list[str], np.ndarray, np.ndarray]:
    """
    从 data_loader 构建带 group 的 lgb.Dataset（训练 / 验证）。
    返回 (train_set, valid_set, feature_names, y_train_raw, y_val_raw)。
    """
    X_tr, y_tr, g_tr = load_training_data(data_dir=data_dir, fill_missing=fill_missing)
    X_va, y_va, g_va = load_validation_data(data_dir=data_dir, fill_missing=fill_missing)

    if sum(g_tr) != len(X_tr) or sum(g_va) != len(X_va):
        raise ValueError(
            f"group 大小之和与样本数不一致: train {sum(g_tr)} vs {len(X_tr)}, "
            f"val {sum(g_va)} vs {len(X_va)}"
        )

    feat_names = list(X_tr.columns)
    X_tr_m = np.ascontiguousarray(X_tr.to_numpy(dtype=np.float32, copy=True))
    X_va_m = np.ascontiguousarray(X_va.to_numpy(dtype=np.float32, copy=True))

    y_tr_rel = future_return_to_relevance(y_tr, g_tr)
    y_va_rel = future_return_to_relevance(y_va, g_va)

    train_set = lgb.Dataset(
        X_tr_m,
        label=y_tr_rel,
        group=g_tr,
        feature_name=feat_names,
        free_raw_data=False,
    )
    valid_set = lgb.Dataset(
        X_va_m,
        label=y_va_rel,
        group=g_va,
        feature_name=feat_names,
        reference=train_set,
        free_raw_data=False,
    )
    return train_set, valid_set, feat_names, y_tr, y_va


def build_datasets_with_label_fn(
    label_fn: Callable[..., np.ndarray],
    *,
    data_dir: Path = DATA_OUT_DIR,
    fill_missing: bool = True,
    label_fn_kwargs: dict[str, Any] | None = None,
) -> tuple[lgb.Dataset, lgb.Dataset, list[str], np.ndarray, np.ndarray]:
    """
    使用自定义标签构造函数构建 lgb.Dataset（训练 / 验证）。

    与 build_datasets 逻辑相同，但支持传入自定义标签函数替代
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
        (train_set, valid_set, feature_names, y_train_raw, y_val_raw)
    """
    X_tr, y_tr, g_tr = load_training_data(data_dir=data_dir, fill_missing=fill_missing)
    X_va, y_va, g_va = load_validation_data(data_dir=data_dir, fill_missing=fill_missing)

    if sum(g_tr) != len(X_tr) or sum(g_va) != len(X_va):
        raise ValueError(
            f"group 大小之和与样本数不一致: train {sum(g_tr)} vs {len(X_tr)}, "
            f"val {sum(g_va)} vs {len(X_va)}"
        )

    feat_names = list(X_tr.columns)

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

    X_tr_m = np.ascontiguousarray(X_tr.to_numpy(dtype=np.float32, copy=True))
    X_va_m = np.ascontiguousarray(X_va.to_numpy(dtype=np.float32, copy=True))

    train_set = lgb.Dataset(
        X_tr_m,
        label=y_tr_rel,
        group=g_tr,
        feature_name=feat_names,
        free_raw_data=False,
    )
    valid_set = lgb.Dataset(
        X_va_m,
        label=y_va_rel,
        group=g_va,
        feature_name=feat_names,
        reference=train_set,
        free_raw_data=False,
    )
    return train_set, valid_set, feat_names, y_tr, y_va


def _ndcg_from_best_score(bst: lgb.Booster, split: str) -> dict[str, float]:
    od = bst.best_score.get(split, {})
    return {
        "ndcg@5": float(od.get("ndcg@5", float("nan"))),
        "ndcg@10": float(od.get("ndcg@10", float("nan"))),
        "ndcg@20": float(od.get("ndcg@20", float("nan"))),
    }


def train_booster(
    params: dict[str, Any],
    train_set: lgb.Dataset,
    valid_set: lgb.Dataset,
    *,
    num_boost_round: int = MAX_BOOST_ROUND,
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
    log_evaluation_period: int = 0,
) -> lgb.Booster:
    """验证集优先用于早停；同时记录训练集 NDCG。"""
    callbacks = [
        lgb.early_stopping(early_stopping_rounds, first_metric_only=True, verbose=False),
        lgb.log_evaluation(log_evaluation_period),
    ]
    return lgb.train(
        params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[valid_set, train_set],
        valid_names=["val", "train"],
        callbacks=callbacks,
    )


@dataclass
class TuneResult:
    best_params: dict[str, Any]
    best_val_ndcg10: float
    n_trials: int


def tune_lightgbm(
    train_set: lgb.Dataset,
    valid_set: lgb.Dataset,
    *,
    base_params: dict[str, Any] | None = None,
    n_trials: int = N_OPTUNA_TRIALS,
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
    seed: int = RANDOM_STATE,
    trials_log_path: Path | None = DEFAULT_TUNE_LOG_PATH,
    logger: logging.Logger | None = None,
) -> TuneResult:
    """
    Optuna 超参搜索：最大化验证集 ndcg@10；每 trial 使用 early_stopping_rounds 早停。
    搜索空间：num_leaves, learning_rate, min_child_samples, feature_fraction。
    """
    base = dict(get_base_params() if base_params is None else base_params)

    def objective(trial: optuna.Trial) -> float:
        params = {
            **base,
            "num_leaves": trial.suggest_int("num_leaves", 10, 63),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 2.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 2.0),
        }
        bst = train_booster(
            params,
            train_set,
            valid_set,
            early_stopping_rounds=early_stopping_rounds,
            log_evaluation_period=0,
        )
        score = float(bst.best_score["val"]["ndcg@10"])
        trial.set_user_attr("best_iteration", int(bst.best_iteration))
        trial.set_user_attr("val_ndcg@5", float(bst.best_score["val"]["ndcg@5"]))
        trial.set_user_attr("val_ndcg@20", float(bst.best_score["val"]["ndcg@20"]))
        if trials_log_path is not None:
            trials_log_path.parent.mkdir(parents=True, exist_ok=True)
            rec = {
                "trial": trial.number,
                "value": score,
                "params": {
                    "num_leaves": params["num_leaves"],
                    "learning_rate": params["learning_rate"],
                    "min_child_samples": params["min_child_samples"],
                    "feature_fraction": params["feature_fraction"],
                    "lambda_l1": params["lambda_l1"],
                    "lambda_l2": params["lambda_l2"],
                },
                "user_attrs": dict(trial.user_attrs),
            }
            with open(trials_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return score

    if trials_log_path is not None and trials_log_path.exists():
        trials_log_path.unlink()

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_trial
    best_params = {
        **base,
        "num_leaves": best.params["num_leaves"],
        "learning_rate": best.params["learning_rate"],
        "min_child_samples": best.params["min_child_samples"],
        "feature_fraction": best.params["feature_fraction"],
        "lambda_l1": best.params["lambda_l1"],
        "lambda_l2": best.params["lambda_l2"],
    }
    msg = (
        f"Optuna 完成: n_trials={n_trials}, best_val_ndcg@10={best.value:.6f}, "
        f"best_params={best.params}"
    )
    if logger:
        logger.info(msg)
    else:
        print(msg)

    return TuneResult(best_params=best_params, best_val_ndcg10=float(best.value), n_trials=n_trials)


def save_feature_importance_json(bst: lgb.Booster, feature_names: list[str], path: Path) -> None:
    imp = bst.feature_importance(importance_type="gain")
    arr = np.asarray(imp, dtype=float)
    payload = {str(n): float(v) for n, v in zip(feature_names, arr)}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def train_final_lightgbm(
    *,
    data_dir: Path = DATA_OUT_DIR,
    model_path: Path = DEFAULT_MODEL_PATH,
    importance_path: Path = DEFAULT_IMPORTANCE_PATH,
    log_path: Path = DEFAULT_LOG_PATH,
    tune: bool = True,
    n_trials: int = N_OPTUNA_TRIALS,
    fill_missing: bool = True,
) -> tuple[lgb.Booster, dict[str, Any]]:
    """
    可选 Optuna 调优后，用最佳参数在全量训练流程上训练并保存模型与特征重要性。
    """
    logger = _setup_file_logger(log_path)
    logger.info("开始 LightGBM LambdaRank：加载数据（fill_missing=%s）", fill_missing)
    train_set, valid_set, feat_names, _y_tr, _y_va = build_datasets(data_dir=data_dir, fill_missing=fill_missing)

    base = get_base_params()
    if tune:
        logger.info("超参数搜索：%d 次 trial，目标=验证集 NDCG@10，early_stopping=%d", n_trials, EARLY_STOPPING_ROUNDS)
        tune_res = tune_lightgbm(
            train_set,
            valid_set,
            base_params=base,
            n_trials=n_trials,
            logger=logger,
        )
        final_params = tune_res.best_params
        logger.info("最优超参: %s", final_params)
        with open(MODELS_DIR / "lightgbm_best_params.json", "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in final_params.items() if isinstance(v, (int, float, str, bool, list))}, f, indent=2)
    else:
        final_params = base
        logger.info("跳过调优，使用基础参数")

    logger.info("使用最终参数训练并早停（early_stopping_rounds=%d）", EARLY_STOPPING_ROUNDS)
    bst = train_booster(
        final_params,
        train_set,
        valid_set,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        log_evaluation_period=50,
    )

    val_m = _ndcg_from_best_score(bst, "val")
    tr_m = _ndcg_from_best_score(bst, "train")
    logger.info(
        "验证集 NDCG: @5=%.6f @10=%.6f @20=%.6f | 训练集 NDCG: @5=%.6f @10=%.6f @20=%.6f | best_iteration=%d",
        val_m["ndcg@5"],
        val_m["ndcg@10"],
        val_m["ndcg@20"],
        tr_m["ndcg@5"],
        tr_m["ndcg@10"],
        tr_m["ndcg@20"],
        bst.best_iteration,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        str(model_path.resolve()).encode("ascii")
        bst.save_model(str(model_path.resolve()))
    except (UnicodeEncodeError, LightGBMError):
        # 非 ASCII 路径或 C API 写盘失败时，使用与 save_model 等价的文本格式写入
        model_path.write_text(bst.model_to_string(), encoding="utf-8")
    logger.info("模型已保存: %s", model_path)

    save_feature_importance_json(bst, feat_names, importance_path)
    logger.info("特征重要性已保存: %s", importance_path)

    meta = {
        "val_ndcg": val_m,
        "train_ndcg": tr_m,
        "best_iteration": int(bst.best_iteration),
        "feature_count": len(feat_names),
        "features": feat_names,
    }
    with open(MODELS_DIR / "lightgbm_metrics.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return bst, final_params


def train_final_lightgbm_with_label_fn(
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
) -> tuple[lgb.Booster, dict[str, Any]]:
    """
    使用自定义标签构造函数训练 LightGBM LambdaRank 模型。

    与 train_final_lightgbm 逻辑相同，但支持传入自定义标签函数。
    模型默认保存为 models/{label_fn_name}_lightgbm.pkl，
    指标保存为 models/{label_fn_name}_lightgbm_metrics.json。

    参数:
        label_fn: 标签构造函数，签名需兼容 (y, group_sizes, **kwargs) -> np.ndarray
        label_fn_name: 标签函数名称，用于生成默认保存路径
        label_fn_kwargs: 传递给 label_fn 的额外关键字参数（如 volatility、sharpe_penalty 等）
        data_dir: 数据目录
        model_path: 模型保存路径，为 None 时使用 models/{label_fn_name}_lightgbm.pkl
        importance_path: 特征重要性保存路径，为 None 时使用 models/{label_fn_name}_lightgbm_feature_importance.json
        log_path: 训练日志路径，为 None 时使用 models/{label_fn_name}_lightgbm_training.log
        tune: 是否进行 Optuna 超参搜索
        n_trials: Optuna 搜索次数
        fill_missing: 是否填充缺失值
        tune_seed: Optuna TPESampler 种子，默认 RANDOM_STATE(42)

    返回:
        (训练好的 Booster, 最终使用的参数字典)
    """
    if model_path is None:
        model_path = MODELS_DIR / f"{label_fn_name}_lightgbm.pkl"
    if importance_path is None:
        importance_path = MODELS_DIR / f"{label_fn_name}_lightgbm_feature_importance.json"
    if log_path is None:
        log_path = MODELS_DIR / f"{label_fn_name}_lightgbm_training.log"

    logger = _setup_file_logger(log_path)
    logger.info(
        "开始 LightGBM LambdaRank（标签函数=%s）：加载数据（fill_missing=%s）",
        label_fn_name, fill_missing,
    )
    train_set, valid_set, feat_names, _y_tr, _y_va = build_datasets_with_label_fn(
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
        tune_res = tune_lightgbm(
            train_set,
            valid_set,
            base_params=base,
            n_trials=n_trials,
            seed=tune_seed,
            logger=logger,
        )
        final_params = tune_res.best_params
        logger.info("最优超参: %s", final_params)
        with open(MODELS_DIR / f"{label_fn_name}_lightgbm_best_params.json", "w", encoding="utf-8") as f:
            json.dump(
                {k: v for k, v in final_params.items() if isinstance(v, (int, float, str, bool, list))},
                f, indent=2,
            )
    else:
        final_params = base
        logger.info("跳过调优，使用基础参数")

    logger.info("使用最终参数训练并早停（early_stopping_rounds=%d）", 150)
    bst = train_booster(
        final_params,
        train_set,
        valid_set,
        early_stopping_rounds=150,
        log_evaluation_period=50,
    )

    val_m = _ndcg_from_best_score(bst, "val")
    tr_m = _ndcg_from_best_score(bst, "train")
    logger.info(
        "验证集 NDCG: @5=%.6f @10=%.6f @20=%.6f | 训练集 NDCG: @5=%.6f @10=%.6f @20=%.6f | best_iteration=%d",
        val_m["ndcg@5"],
        val_m["ndcg@10"],
        val_m["ndcg@20"],
        tr_m["ndcg@5"],
        tr_m["ndcg@10"],
        tr_m["ndcg@20"],
        bst.best_iteration,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        str(model_path.resolve()).encode("ascii")
        bst.save_model(str(model_path.resolve()))
    except (UnicodeEncodeError, LightGBMError):
        model_path.write_text(bst.model_to_string(), encoding="utf-8")
    logger.info("模型已保存: %s", model_path)

    save_feature_importance_json(bst, feat_names, importance_path)
    logger.info("特征重要性已保存: %s", importance_path)

    meta = {
        "label_fn_name": label_fn_name,
        "val_ndcg": val_m,
        "train_ndcg": tr_m,
        "best_iteration": int(bst.best_iteration),
        "feature_count": len(feat_names),
        "features": feat_names,
    }
    metrics_path = MODELS_DIR / f"{label_fn_name}_lightgbm_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return bst, final_params


def load_lightgbm_model(path: Path | None = None) -> lgb.Booster:
    p = path or DEFAULT_MODEL_PATH
    if not p.exists():
        raise FileNotFoundError(str(p))
    try:
        str(p.resolve()).encode("ascii")
        return lgb.Booster(model_file=str(p.resolve()))
    except (UnicodeEncodeError, LightGBMError):
        return lgb.Booster(model_str=p.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LightGBM LambdaRank with optional Optuna tuning.")
    parser.add_argument("--trials", type=int, default=N_OPTUNA_TRIALS, help="Optuna trials (default 20)")
    parser.add_argument("--no-tune", action="store_true", help="Skip hyperparameter search, use base params only")
    args = parser.parse_args()
    train_final_lightgbm(tune=not args.no_tune, n_trials=max(1, args.trials))


if __name__ == "__main__":
    main()
