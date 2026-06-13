"""FastAPI 依赖：单例 ModelPredictor，避免每次请求重复加载模型。"""
from __future__ import annotations

import logging
from functools import lru_cache

from src.config import get_settings
from src.predictor import ModelKind, ModelPredictor

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _predictor_singleton(model_key: str) -> ModelPredictor:
    mt: ModelKind = "lightgbm" if model_key == "lightgbm" else "xgboost"
    return ModelPredictor(mt, data_dir="/app/data")


def get_predictor() -> ModelPredictor:
    s = get_settings()
    key = (s.default_predict_model or "lightgbm").strip().lower()
    if key not in ("lightgbm", "xgboost"):
        key = "lightgbm"
    return _predictor_singleton(key)


def get_predictor_by_type(model_type: str) -> ModelPredictor:
    """根据指定的模型类型获取 ModelPredictor 实例。

    入参:
        model_type: 模型类型字符串，预期为 "lightgbm" 或 "xgboost"。

    出参:
        对应模型类型的 ModelPredictor 实例。

    异常:
        若输入不合法，记录警告日志并回退到 "lightgbm"。
    """
    key = (model_type or "lightgbm").strip().lower()
    if key not in ("lightgbm", "xgboost"):
        logger.warning("未知的模型类型 '%s'，回退到 lightgbm", model_type)
        key = "lightgbm"
    return _predictor_singleton(key)


def clear_predictor_cache() -> None:
    """单测或热切换模型时清空缓存。"""
    _predictor_singleton.cache_clear()
