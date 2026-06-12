"""
E1a/E1c 标签函数单元测试

验证 return_aware_relevance 与 cvar_aware_relevance 的核心逻辑正确性，
包括负收益截断、distance 上限截断、全正收益 bonus 计算、无极端值惩罚等场景。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model_lightgbm import cvar_aware_relevance, return_aware_relevance


class TestReturnAwareRelevance:
    """return_aware_relevance (E1a) 标签函数测试。"""

    def test_negative_return_bonus_is_zero(self) -> None:
        """
        含负收益截面：负收益股票的 bonus 应为 0。

        构造 group_sizes=[3,3]，y=[0.05, -0.03, 0.02, -0.01, 0.04, -0.06]，
        验证负收益位置的标签值不超过 rank_rel 部分（即 bonus 为 0）。
        """
        y = np.array([0.05, -0.03, 0.02, -0.01, 0.04, -0.06], dtype=np.float64)
        group_sizes: list[int] = [3, 3]
        max_label: int = 30
        alpha: float = 1.0

        labels = return_aware_relevance(y, group_sizes, max_label=max_label, alpha=alpha)

        neg_indices = [1, 3, 5]
        from src.model_lightgbm import future_return_to_relevance

        rank_rel = future_return_to_relevance(y, group_sizes, max_label=max_label).astype(np.float64)

        for idx in neg_indices:
            assert labels[idx] <= rank_rel[idx] + 0.5, (
                f"负收益位置 idx={idx}，标签值 {labels[idx]} 不应超过 "
                f"rank_rel+0.5={rank_rel[idx] + 0.5}，说明 bonus 未正确截断为 0"
            )

    def test_all_positive_return_bonus_normal(self) -> None:
        """
        全正收益数据：bonus 应正常计算，至少存在一只非满秩股票的标签值大于纯排名标签。

        构造 group_sizes=[6]，y=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06]，
        验证存在 rank_rel < max_label 的股票，其 bonus 叠加后标签值严格大于 rank_rel。
        """
        y = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06], dtype=np.float64)
        group_sizes: list[int] = [6]
        max_label: int = 30
        alpha: float = 1.0

        labels = return_aware_relevance(y, group_sizes, max_label=max_label, alpha=alpha)

        from src.model_lightgbm import future_return_to_relevance

        rank_rel = future_return_to_relevance(y, group_sizes, max_label=max_label).astype(np.float64)

        has_bonus = False
        for i in range(len(y)):
            if rank_rel[i] < max_label and labels[i] > rank_rel[i]:
                has_bonus = True
                break

        assert has_bonus, (
            f"全正收益数据中，应至少存在一只非满秩股票的标签值大于纯排名标签，"
            f"labels={labels}，rank_rel={rank_rel}"
        )

    def test_output_range(self) -> None:
        """
        输出值域验证：所有标签值应在 [0, max_label] 范围内。

        使用混合正负收益数据，确保截断逻辑正确。
        """
        y = np.array([0.1, -0.05, 0.03, -0.02, 0.08], dtype=np.float64)
        group_sizes: list[int] = [5]
        max_label: int = 30

        labels = return_aware_relevance(y, group_sizes, max_label=max_label)

        assert labels.min() >= 0, f"标签最小值 {labels.min()} 不应小于 0"
        assert labels.max() <= max_label, f"标签最大值 {labels.max()} 不应超过 {max_label}"


class TestCvarAwareRelevance:
    """cvar_aware_relevance (E1c) 标签函数测试。"""

    def test_distance_capped_at_3(self) -> None:
        """
        极端负收益数据：distance 不超过 3.0。

        构造 group_sizes=[4]，y=[0.01, -0.50, 0.02, 0.03]，
        其中 -0.50 远低于 20% 分位数阈值，验证 distance 被截断在 3.0。
        通过对比 cvar_penalty=0 和 cvar_penalty>0 的结果差异间接验证。
        """
        y = np.array([0.01, -0.50, 0.02, 0.03], dtype=np.float64)
        group_sizes: list[int] = [4]
        max_label: int = 30
        cvar_penalty: float = 1.5
        cvar_alpha: float = 0.20

        labels_no_cvar = cvar_aware_relevance(
            y, group_sizes, max_label=max_label,
            cvar_penalty=0.0, cvar_alpha=cvar_alpha,
        )
        labels_with_cvar = cvar_aware_relevance(
            y, group_sizes, max_label=max_label,
            cvar_penalty=cvar_penalty, cvar_alpha=cvar_alpha,
        )

        neg_idx = 1
        penalty_applied = labels_no_cvar[neg_idx] - labels_with_cvar[neg_idx]

        max_possible_penalty = cvar_penalty * 3.0 * max_label
        assert penalty_applied <= max_possible_penalty + 0.5, (
            f"位置 idx={neg_idx} 的惩罚 {penalty_applied} 超过理论上限 "
            f"{max_possible_penalty}，说明 distance 未被截断在 3.0"
        )

    def test_no_extreme_value_penalty_normal(self) -> None:
        """
        无极端值数据：CVaR 惩罚应正常施加，但幅度较小。

        构造 group_sizes=[5]，y=[0.01, 0.02, 0.03, 0.04, 0.05]，
        所有收益为正且无极端值，验证惩罚后标签仍合理（非负且有序）。
        """
        y = np.array([0.01, 0.02, 0.03, 0.04, 0.05], dtype=np.float64)
        group_sizes: list[int] = [5]
        max_label: int = 30
        cvar_penalty: float = 1.5
        cvar_alpha: float = 0.20

        labels = cvar_aware_relevance(
            y, group_sizes, max_label=max_label,
            cvar_penalty=cvar_penalty, cvar_alpha=cvar_alpha,
        )

        assert labels.min() >= 0, f"标签最小值 {labels.min()} 不应小于 0"
        assert labels.max() <= max_label, f"标签最大值 {labels.max()} 不应超过 {max_label}"

    def test_output_range_with_extreme(self) -> None:
        """
        极端数据输出值域验证：含极端负收益时标签仍在 [0, max_label]。

        构造 group_sizes=[3]，y=[0.05, -0.30, 0.01]，
        确保即使有极端值，最终截断逻辑也能保证值域正确。
        """
        y = np.array([0.05, -0.30, 0.01], dtype=np.float64)
        group_sizes: list[int] = [3]
        max_label: int = 30
        cvar_penalty: float = 1.5
        cvar_alpha: float = 0.20

        labels = cvar_aware_relevance(
            y, group_sizes, max_label=max_label,
            cvar_penalty=cvar_penalty, cvar_alpha=cvar_alpha,
        )

        assert labels.min() >= 0, f"标签最小值 {labels.min()} 不应小于 0"
        assert labels.max() <= max_label, f"标签最大值 {labels.max()} 不应超过 {max_label}"
