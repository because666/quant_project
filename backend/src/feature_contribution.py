"""
特征贡献分解模块：使用 LightGBM predict_contrib 计算每只股票的 SHAP 近似贡献值，
用于实验3中 E3b 粒度的个股因子贡献分解展示。
"""
from __future__ import annotations

import logging
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_feature_contribution(
    model: lgb.Booster,
    panel_df: pd.DataFrame,
    factor_cols: list[str],
    top_k: int = 5,
) -> dict[str, list[tuple[str, float]]]:
    """
    使用LightGBM predict_contrib计算每只股票的Top-K因子贡献分解。

    参数:
        model: 训练好的LightGBM Booster对象
        panel_df: 截面数据，需包含 stock_code 列和因子列
        factor_cols: 因子列名列表
        top_k: 每只股票返回Top-K个贡献最大的因子

    返回:
        字典，key为stock_code，value为[(factor_name, contribution_value), ...]列表，
        按贡献值绝对值降序排列

    异常:
        ValueError: panel_df为空或缺少必要列时抛出
        RuntimeError: predict_contrib调用失败时抛出
    """
    if panel_df.empty:
        raise ValueError("panel_df 为空，无法计算特征贡献")

    if "stock_code" not in panel_df.columns:
        raise ValueError("panel_df 必须包含 stock_code 列")

    missing_cols = [c for c in factor_cols if c not in panel_df.columns]
    if missing_cols:
        raise ValueError(f"panel_df 缺少因子列: {missing_cols}")

    stock_codes = panel_df["stock_code"].astype(str).to_numpy()
    X = panel_df[factor_cols].to_numpy(dtype=np.float32, copy=True)

    nan_count = int(np.isnan(X).sum())
    if nan_count > 0:
        logger.warning("因子矩阵含 %d 个缺失值，将以列中位数填充", nan_count)
        col_medians = np.nanmedian(X, axis=0)
        nan_mask = np.isnan(X)
        for j in range(X.shape[1]):
            if np.isnan(col_medians[j]):
                col_medians[j] = 0.0
            X[nan_mask[:, j], j] = col_medians[j]

    try:
        contrib_matrix = model.predict(X, pred_contrib=True)
    except Exception as exc:
        raise RuntimeError(f"LightGBM predict_contrib 调用失败: {exc}") from exc

    contrib_arr = np.asarray(contrib_matrix, dtype=np.float64)
    if contrib_arr.ndim == 1:
        raise RuntimeError(
            f"predict_contrib 返回一维数组，期望二维 (n_samples, n_features+1)，"
            f"实际形状: {contrib_arr.shape}"
        )

    n_features = contrib_arr.shape[1] - 1
    if n_features != len(factor_cols):
        logger.warning(
            "predict_contrib 返回 %d 个特征列，但 factor_cols 有 %d 个；"
            "将使用模型特征名对齐",
            n_features,
            len(factor_cols),
        )
        model_feature_names = model.feature_name()
        if model_feature_names and len(model_feature_names) == n_features:
            factor_names = [str(n) for n in model_feature_names]
        else:
            factor_names = [f"f{i}" for i in range(n_features)]
    else:
        factor_names = list(factor_cols)

    result: dict[str, list[tuple[str, float]]] = {}
    for i in range(contrib_arr.shape[0]):
        code = str(stock_codes[i])
        contributions = [(factor_names[j], float(contrib_arr[i, j])) for j in range(n_features)]
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        result[code] = contributions[:top_k]

    return result
