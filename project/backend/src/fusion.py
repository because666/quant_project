"""
多模型融合模块：提供分数平均、RRF、Stacking、加权RRF四种融合策略，
以及 FusionPredictor 封装类，提供与 ModelPredictor 一致的 predict 接口。

用法::

    from src.fusion import FusionPredictor
    fp = FusionPredictor("average", ["lightgbm", "xgboost"])
    result = fp.predict(panel_df)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .predictor import ModelPredictor

logger = logging.getLogger(__name__)


def score_average_fusion(
    predictions_list: list[np.ndarray],
    stock_codes: list[str],
) -> pd.DataFrame:
    """
    E2a：分数平均融合。将多个模型的预测分数归一化后取平均。

    参数:
        predictions_list: 多个模型的预测分数数组列表，每个形状 (N,)
        stock_codes: 股票代码列表，长度 N

    返回:
        DataFrame，包含 stock_code 和 score 列，按 score 降序排列

    异常:
        ValueError: predictions_list 为空或长度与 stock_codes 不一致时抛出
    """
    if not predictions_list:
        raise ValueError("predictions_list 不能为空")
    n = len(stock_codes)
    for i, pred in enumerate(predictions_list):
        if len(pred) != n:
            raise ValueError(
                f"predictions_list[{i}] 长度 {len(pred)} 与 stock_codes 长度 {n} 不一致"
            )

    normalized_list: list[np.ndarray] = []
    for pred in predictions_list:
        arr = np.asarray(pred, dtype=np.float64)
        min_val = arr.min()
        max_val = arr.max()
        if max_val - min_val < 1e-12:
            normalized = np.zeros_like(arr)
        else:
            normalized = (arr - min_val) / (max_val - min_val)
        normalized_list.append(normalized)

    avg_score = np.mean(normalized_list, axis=0)

    result = pd.DataFrame({
        "stock_code": stock_codes,
        "score": avg_score,
    })
    return result.sort_values("score", ascending=False, kind="mergesort").reset_index(drop=True)


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    k: int = 60,
    weights: list[float] | None = None,
) -> tuple[list[str], dict[str, float]]:
    """
    E2b：Reciprocal Rank Fusion 排序融合。

    参数:
        rankings: 多个模型的排名列表，每个元素为股票代码的有序列表（按分数降序）
        k: 平滑常数，默认 60
        weights: 各模型的权重，默认等权

    返回:
        (融合后的股票代码排序列表, RRF分数字典)

    异常:
        ValueError: rankings 为空或 weights 长度与 rankings 不一致时抛出
    """
    if not rankings:
        raise ValueError("rankings 不能为空")
    n_models = len(rankings)
    if weights is None:
        weights = [1.0] * n_models
    elif len(weights) != n_models:
        raise ValueError(
            f"weights 长度 {len(weights)} 与 rankings 长度 {n_models} 不一致"
        )

    rrf_scores: dict[str, float] = {}
    for model_idx, ranking in enumerate(rankings):
        w = weights[model_idx]
        for rank_pos, code in enumerate(ranking):
            score = w / (k + rank_pos + 1)
            rrf_scores[code] = rrf_scores.get(code, 0.0) + score

    sorted_codes = sorted(rrf_scores.keys(), key=lambda c: rrf_scores[c], reverse=True)
    return sorted_codes, rrf_scores


def stacking_fusion(
    val_predictions_list: list[np.ndarray],
    val_y: np.ndarray,
    val_group_sizes: list[int],
    base_nDCG_fn: Callable | None = None,
) -> tuple[Ridge, dict[str, Any]]:
    """
    E2c：Stacking融合。使用Ridge元学习器在验证集上学习最优融合权重。

    参数:
        val_predictions_list: 各模型在验证集上的预测分数列表
        val_y: 验证集标签
        val_group_sizes: 验证集截面大小列表
        base_nDCG_fn: 可选的NDCG计算函数，用于记录元信息

    返回:
        (训练好的Ridge模型, 元信息字典)

    异常:
        ValueError: val_predictions_list 为空或数据长度不一致时抛出
    """
    if not val_predictions_list:
        raise ValueError("val_predictions_list 不能为空")

    n_samples = len(val_y)
    for i, pred in enumerate(val_predictions_list):
        if len(pred) != n_samples:
            raise ValueError(
                f"val_predictions_list[{i}] 长度 {len(pred)} 与 val_y 长度 {n_samples} 不一致"
            )

    X_meta = np.column_stack(val_predictions_list)
    y_meta = np.asarray(val_y, dtype=np.float64)

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_meta, y_meta)

    meta_info: dict[str, Any] = {
        "n_models": len(val_predictions_list),
        "n_samples": n_samples,
        "n_groups": len(val_group_sizes),
        "coefficients": ridge.coef_.tolist(),
        "intercept": float(ridge.intercept_),
    }

    if base_nDCG_fn is not None:
        try:
            stacked_pred = ridge.predict(X_meta)
            ndcg_val = base_nDCG_fn(val_y, stacked_pred, val_group_sizes)
            meta_info["stacked_ndcg"] = float(ndcg_val)
        except Exception as exc:
            logger.warning("Stacking NDCG 计算失败: %s", exc)

    return ridge, meta_info


def weighted_rrf_fusion(
    rankings: list[list[str]],
    ndcg_weights: list[float],
    k: int = 60,
) -> tuple[list[str], dict[str, float]]:
    """
    E2d：加权RRF融合。根据验证集NDCG@10为各模型分配权重。

    参数:
        rankings: 多个模型的排名列表
        ndcg_weights: 各模型的NDCG@10权重
        k: 平滑常数，默认 60

    返回:
        (融合后的股票代码排序列表, RRF分数字典)

    异常:
        ValueError: rankings 为空或 ndcg_weights 长度与 rankings 不一致时抛出
    """
    if not rankings:
        raise ValueError("rankings 不能为空")
    if len(ndcg_weights) != len(rankings):
        raise ValueError(
            f"ndcg_weights 长度 {len(ndcg_weights)} 与 rankings 长度 {len(rankings)} 不一致"
        )

    return reciprocal_rank_fusion(rankings, k=k, weights=ndcg_weights)


def _scores_to_ranking(scores: np.ndarray, stock_codes: list[str]) -> list[str]:
    """
    将分数数组转为排名列表（按分数降序排列的股票代码列表）。

    参数:
        scores: 预测分数数组，形状 (N,)
        stock_codes: 股票代码列表，长度 N

    返回:
        按分数降序排列的股票代码列表
    """
    arr = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-arr)
    return [stock_codes[i] for i in order]


def _ranking_to_scores(ranking: list[str]) -> tuple[list[str], np.ndarray]:
    """
    将排名列表转为分数：排名越前分数越高，使用逆序数作为分数。

    参数:
        ranking: 按排名排列的股票代码列表

    返回:
        (股票代码列表, 分数数组)
    """
    n = len(ranking)
    scores = np.array([n - i for i in range(n)], dtype=np.float64)
    return ranking, scores


class FusionPredictor:
    """
    融合预测器，封装多模型融合逻辑，提供与 ModelPredictor 一致的 predict 接口。
    """

    def __init__(
        self,
        fusion_type: str,
        model_types: list[str],
        data_dir: Path | None = None,
        k: int = 60,
        weights: list[float] | None = None,
        ridge_model: Ridge | None = None,
    ) -> None:
        """
        初始化融合预测器。

        参数:
            fusion_type: 融合类型，"average"/"rrf"/"stacking"/"weighted_rrf"
            model_types: 参与融合的模型类型列表，如 ["lightgbm", "xgboost"]
            data_dir: 数据目录
            k: RRF平滑常数
            weights: 加权RRF的权重
            ridge_model: Stacking的Ridge模型

        异常:
            ValueError: fusion_type 不合法或 model_types 为空时抛出
            FileNotFoundError: 子模型文件不存在时抛出
        """
        valid_types = {"average", "rrf", "stacking", "weighted_rrf"}
        if fusion_type not in valid_types:
            raise ValueError(f"fusion_type 必须为 {valid_types} 之一，收到 {fusion_type!r}")
        if not model_types:
            raise ValueError("model_types 不能为空")

        self.fusion_type: str = fusion_type
        self.model_types: list[str] = model_types
        self.k: int = k
        self.weights: list[float] | None = weights
        self.ridge_model: Ridge | None = ridge_model

        self._predictors: list[ModelPredictor] = []
        for mt in model_types:
            predictor = ModelPredictor(mt, data_dir=data_dir)
            self._predictors.append(predictor)

        self._factor_cols: list[str] = self._predictors[0]._factor_cols

    def predict(self, panel_df: pd.DataFrame) -> pd.DataFrame:
        """
        对截面数据执行融合预测。

        参数:
            panel_df: 截面数据，需包含 stock_code 列和因子列

        返回:
            DataFrame，包含 stock_code 和 score 列，按 score 降序排列

        异常:
            ValueError: panel_df 为空或缺少必要列时抛出
            RuntimeError: 融合计算失败时抛出
        """
        if panel_df.empty:
            raise ValueError("panel_df 为空")
        if "stock_code" not in panel_df.columns:
            raise ValueError("panel_df 必须包含 stock_code 列")

        stock_codes: list[str] = panel_df["stock_code"].astype(str).tolist()

        predictions_list: list[np.ndarray] = []
        for predictor in self._predictors:
            pred_df = predictor.predict(panel_df)
            score_map = dict(zip(
                pred_df["stock_code"].astype(str).to_numpy(),
                pred_df["score"].to_numpy(dtype=np.float64),
            ))
            scores = np.array([score_map.get(code, 0.0) for code in stock_codes], dtype=np.float64)
            predictions_list.append(scores)

        if self.fusion_type == "average":
            return score_average_fusion(predictions_list, stock_codes)

        if self.fusion_type in ("rrf", "weighted_rrf", "stacking"):
            rankings: list[list[str]] = []
            for scores in predictions_list:
                ranking = _scores_to_ranking(scores, stock_codes)
                rankings.append(ranking)

            if self.fusion_type == "rrf":
                fused_ranking, rrf_score_map = reciprocal_rank_fusion(rankings, k=self.k)
            elif self.fusion_type == "weighted_rrf":
                if self.weights is None:
                    raise ValueError("weighted_rrf 融合需要提供 weights 参数")
                fused_ranking, rrf_score_map = weighted_rrf_fusion(rankings, self.weights, k=self.k)
            else:
                if self.ridge_model is None:
                    raise ValueError("stacking 融合需要提供 ridge_model 参数")
                X_meta = np.column_stack(predictions_list)
                stacked_scores = self.ridge_model.predict(X_meta)
                code_score_map = dict(zip(stock_codes, stacked_scores))
                sorted_codes = sorted(stock_codes, key=lambda c: code_score_map.get(c, 0.0), reverse=True)
                result = pd.DataFrame({
                    "stock_code": sorted_codes,
                    "score": [code_score_map[c] for c in sorted_codes],
                })
                return result

            fused_scores = np.array([rrf_score_map.get(code, 0.0) for code in stock_codes], dtype=np.float64)
            result = pd.DataFrame({
                "stock_code": stock_codes,
                "score": fused_scores,
            })
            return result.sort_values("score", ascending=False, kind="mergesort").reset_index(drop=True)

        raise ValueError(f"不支持的融合类型: {self.fusion_type!r}")

    def predict_panel(self, factor_df: pd.DataFrame) -> pd.DataFrame:
        """
        多调仓日批量推理：每行含 date、stock_code 及因子列；
        返回与输入行对齐的 date、stock_code、score。

        参数:
            factor_df: 截面数据，需包含 date、stock_code 列和因子列

        返回:
            DataFrame，包含 date、stock_code、score 列

        异常:
            ValueError: factor_df 为空或缺少必要列时抛出
        """
        if factor_df.empty:
            return pd.DataFrame(columns=["date", "stock_code", "score"])
        if "date" not in factor_df.columns:
            raise ValueError("predict_panel 需要 date 列")

        work = factor_df.reset_index(drop=True)
        dt_arr = pd.to_datetime(work["date"], errors="coerce")

        parts: list[pd.DataFrame] = []
        for dt, group in work.groupby("date", sort=True):
            try:
                pred_df = self.predict(group)
                parts.append(pd.DataFrame({
                    "date": pd.Timestamp(dt),
                    "stock_code": pred_df["stock_code"].astype(str).to_numpy(),
                    "score": pred_df["score"].to_numpy(dtype=np.float64),
                }))
            except Exception as exc:
                logger.warning("融合预测失败（日期=%s）: %s", dt, exc)

        if not parts:
            return pd.DataFrame(columns=["date", "stock_code", "score"])
        return pd.concat(parts, ignore_index=True)

    def __repr__(self) -> str:
        return (
            f"FusionPredictor(fusion_type={self.fusion_type!r}, "
            f"model_types={self.model_types!r})"
        )
