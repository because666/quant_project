"""
统计检验+稳健性检验单元测试
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics import bootstrap_metric, paired_significance_test, sharpe_ratio


class TestDeflatedSharpeRatio:
    """Deflated Sharpe Ratio测试。"""

    def test_basic_dsr(self) -> None:
        """基本DSR计算应返回0-1之间的值。"""
        from scripts.run_statistical_tests import deflated_sharpe_ratio
        dsr = deflated_sharpe_ratio(
            sharpe_annual=1.5, n_trials=20, skew=0.0, kurtosis=0.0, n_obs=100,
        )
        assert 0.0 <= dsr <= 1.0

    def test_high_sharpe_should_be_credible(self) -> None:
        """高夏普比率+大样本应通过DSR检验。"""
        from scripts.run_statistical_tests import deflated_sharpe_ratio
        dsr = deflated_sharpe_ratio(
            sharpe_annual=3.0, n_trials=5, skew=0.0, kurtosis=0.0, n_obs=500,
        )
        assert dsr > 0.5

    def test_low_sharpe_should_not_be_credible(self) -> None:
        """低夏普比率应不通过DSR检验。"""
        from scripts.run_statistical_tests import deflated_sharpe_ratio
        dsr = deflated_sharpe_ratio(
            sharpe_annual=0.1, n_trials=20, skew=0.0, kurtosis=0.0, n_obs=50,
        )
        assert dsr < 0.5

    def test_zero_trials(self) -> None:
        """0次试验应返回0。"""
        from scripts.run_statistical_tests import deflated_sharpe_ratio
        dsr = deflated_sharpe_ratio(
            sharpe_annual=1.0, n_trials=0, skew=0.0, kurtosis=0.0, n_obs=100,
        )
        assert dsr == 0.0

    def test_insufficient_obs(self) -> None:
        """观测数不足应返回0。"""
        from scripts.run_statistical_tests import deflated_sharpe_ratio
        dsr = deflated_sharpe_ratio(
            sharpe_annual=1.0, n_trials=20, skew=0.0, kurtosis=0.0, n_obs=1,
        )
        assert dsr == 0.0

    def test_negative_sharpe(self) -> None:
        """负夏普比率应返回接近0的DSR。"""
        from scripts.run_statistical_tests import deflated_sharpe_ratio
        dsr = deflated_sharpe_ratio(
            sharpe_annual=-1.0, n_trials=20, skew=0.0, kurtosis=0.0, n_obs=100,
        )
        assert dsr < 0.01

    def test_more_trials_reduces_dsr(self) -> None:
        """更多试验次数应降低DSR（多重试验惩罚更强）。"""
        from scripts.run_statistical_tests import deflated_sharpe_ratio
        dsr_5 = deflated_sharpe_ratio(
            sharpe_annual=1.0, n_trials=5, skew=0.0, kurtosis=0.0, n_obs=100,
        )
        dsr_100 = deflated_sharpe_ratio(
            sharpe_annual=1.0, n_trials=100, skew=0.0, kurtosis=0.0, n_obs=100,
        )
        assert dsr_5 > dsr_100


class TestSigMark:
    """显著性标记函数测试。"""

    def test_p01(self) -> None:
        """p<0.01应标记***。"""
        from scripts.run_statistical_tests import _sig_mark
        assert _sig_mark(0.005) == "***"

    def test_p05(self) -> None:
        """p<0.05应标记**。"""
        from scripts.run_statistical_tests import _sig_mark
        assert _sig_mark(0.03) == "**"

    def test_p10(self) -> None:
        """p<0.1应标记*。"""
        from scripts.run_statistical_tests import _sig_mark
        assert _sig_mark(0.08) == "*"

    def test_p_large(self) -> None:
        """p>=0.1应无标记。"""
        from scripts.run_statistical_tests import _sig_mark
        assert _sig_mark(0.5) == ""

    def test_nan(self) -> None:
        """NaN应返回空字符串。"""
        from scripts.run_statistical_tests import _sig_mark
        assert _sig_mark(float("nan")) == ""


class TestSafeFloat:
    """_safe_float函数测试。"""

    def test_normal(self) -> None:
        from scripts.run_statistical_tests import _safe_float
        assert _safe_float(3.14) == 3.14

    def test_none(self) -> None:
        from scripts.run_statistical_tests import _safe_float
        assert _safe_float(None) == 0.0

    def test_nan(self) -> None:
        from scripts.run_statistical_tests import _safe_float
        assert _safe_float(float("nan")) == 0.0

    def test_inf(self) -> None:
        from scripts.run_statistical_tests import _safe_float
        assert _safe_float(float("inf")) == 0.0


class TestGetMethodConfigs:
    """方法配置测试。"""

    def test_configs_exist(self) -> None:
        from scripts.run_statistical_tests import get_method_configs
        configs = get_method_configs()
        assert len(configs) >= 4
        assert "B-LGBM" in configs
        assert "B-XGB" in configs

    def test_each_has_required_keys(self) -> None:
        from scripts.run_statistical_tests import get_method_configs
        configs = get_method_configs()
        for name, config in configs.items():
            assert "label" in config
            assert "predictor_type" in config
            assert "vol_penalty" in config


class TestOutputFiles:
    """输出文件完整性测试。"""

    DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "statistical_tests"

    def test_report_exists(self) -> None:
        path = self.DATA_DIR / "report.md"
        assert path.exists(), f"缺少: {path}"
        content = path.read_text(encoding="utf-8")
        assert "Bootstrap" in content
        assert "配对" in content
        assert "Deflated Sharpe" in content
        assert "Top N" in content
        assert "持有期" in content
        assert "RRF" in content

    def test_bootstrap_ci_json(self) -> None:
        path = self.DATA_DIR / "bootstrap_ci.json"
        assert path.exists(), f"缺少: {path}"
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) >= 2
        for name, ci in data.items():
            assert "sharpe_ratio" in ci
            assert "point" in ci["sharpe_ratio"]

    def test_paired_tests_json(self) -> None:
        path = self.DATA_DIR / "paired_tests.json"
        assert path.exists(), f"缺少: {path}"
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) >= 1

    def test_deflated_sharpe_json(self) -> None:
        path = self.DATA_DIR / "deflated_sharpe.json"
        assert path.exists(), f"缺少: {path}"
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) >= 2
        for name, d in data.items():
            assert "dsr" in d
            assert "sharpe_ratio" in d

    def test_sensitivity_topn_json(self) -> None:
        path = self.DATA_DIR / "sensitivity_topn.json"
        assert path.exists(), f"缺少: {path}"
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) >= 2

    def test_sensitivity_holding_json(self) -> None:
        path = self.DATA_DIR / "sensitivity_holding.json"
        assert path.exists(), f"缺少: {path}"
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) >= 2

    def test_sensitivity_rrf_k_json(self) -> None:
        path = self.DATA_DIR / "sensitivity_rrf_k.json"
        assert path.exists(), f"缺少: {path}"
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) >= 2
