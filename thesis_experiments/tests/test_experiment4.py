"""
实验4单元测试：综合对比 + 分年度分析 + 多空组合 + 统计检验
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

from src.metrics import (
    bootstrap_metric,
    metrics_by_year,
    paired_significance_test,
    quantile_portfolio_returns,
    sharpe_ratio,
)


class TestYearlyAnalysis:
    """分年度分析测试。"""

    def test_metrics_by_year_basic(self) -> None:
        """基本分年度分析应返回正确年份数和指标。"""
        dates = pd.date_range("2022-01-07", periods=104, freq="W-FRI")
        nav = pd.Series(np.linspace(1.0, 1.5, 104), index=dates)
        result = metrics_by_year(nav)
        assert len(result) >= 2
        assert "annualized_return" in result.columns
        assert "sharpe_ratio" in result.columns
        assert "max_drawdown" in result.columns

    def test_metrics_by_year_single_year(self) -> None:
        """单年数据应返回1行。"""
        dates = pd.date_range("2023-01-07", periods=52, freq="W-FRI")
        nav = pd.Series(np.linspace(1.0, 1.2, 52), index=dates)
        result = metrics_by_year(nav)
        assert len(result) == 1

    def test_metrics_by_year_empty(self) -> None:
        """空数据应返回空DataFrame。"""
        nav = pd.Series([], dtype=float)
        result = metrics_by_year(nav)
        assert result.empty


class TestQuantilePortfolioReturns:
    """多空组合收益测试。"""

    def test_five_quantiles(self) -> None:
        """5分位数分组应返回Q1-Q5和LS。"""
        n_dates = 10
        n_stocks = 100
        rows = []
        for d in range(n_dates):
            for s in range(n_stocks):
                rows.append({
                    "date": f"2023-{d+1:02d}-01",
                    "stock_code": f"{s:06d}",
                    "score": float(n_stocks - s + np.random.randn() * 0.1),
                    "future_return_1w": np.random.randn() * 0.02,
                })
        df = pd.DataFrame(rows)
        result = quantile_portfolio_returns(df, n_quantiles=5)
        assert "Q1" in result
        assert "Q5" in result
        assert "LS" in result
        assert len(result["Q1"]) == n_dates
        assert len(result["LS"]) == n_dates

    def test_empty_dataframe(self) -> None:
        """空DataFrame应返回空字典。"""
        df = pd.DataFrame(columns=["date", "stock_code", "score", "future_return_1w"])
        result = quantile_portfolio_returns(df)
        assert result == {}

    def test_missing_columns(self) -> None:
        """缺少必要列应返回空字典。"""
        df = pd.DataFrame({"date": [], "stock_code": []})
        result = quantile_portfolio_returns(df)
        assert result == {}

    def test_long_short_calculation(self) -> None:
        """多空收益应等于Q1-Q5。"""
        np.random.seed(42)
        n_dates = 20
        n_stocks = 50
        rows = []
        for d in range(n_dates):
            for s in range(n_stocks):
                rows.append({
                    "date": f"2023-{(d % 12) + 1:02d}-01",
                    "stock_code": f"{s:06d}",
                    "score": float(n_stocks - s),
                    "future_return_1w": np.random.randn() * 0.03,
                })
        df = pd.DataFrame(rows)
        result = quantile_portfolio_returns(df, n_quantiles=5)
        ls_manual = result["Q1"] - result["Q5"]
        np.testing.assert_array_almost_equal(result["LS"], ls_manual)


class TestBootstrapMetric:
    """Bootstrap置信区间测试。"""

    def test_basic_ci(self) -> None:
        """基本Bootstrap应返回point/lower/upper。"""
        returns = np.random.randn(100) * 0.02
        result = bootstrap_metric(returns, sharpe_ratio, n_bootstrap=100)
        assert "point" in result
        assert "lower" in result
        assert "upper" in result
        assert result["lower"] < result["point"] < result["upper"]

    def test_short_series(self) -> None:
        """过短序列应返回NaN。"""
        returns = np.array([0.01])
        result = bootstrap_metric(returns, sharpe_ratio, n_bootstrap=10)
        assert np.isnan(result["point"])

    def test_deterministic_with_seed(self) -> None:
        """相同种子应产生相同结果。"""
        returns = np.random.randn(50) * 0.02
        r1 = bootstrap_metric(returns, sharpe_ratio, n_bootstrap=50, seed=42)
        r2 = bootstrap_metric(returns, sharpe_ratio, n_bootstrap=50, seed=42)
        assert abs(r1["point"] - r2["point"]) < 1e-10


class TestPairedSignificanceTest:
    """配对显著性检验测试。"""

    def test_identical_series(self) -> None:
        """非常接近的序列应不显著。"""
        np.random.seed(42)
        a = np.random.randn(50)
        b = a + np.random.randn(50) * 1e-8
        result = paired_significance_test(a, b)
        assert result["p_value"] > 0.01

    def test_different_series(self) -> None:
        """显著不同的序列应检测出差异。"""
        np.random.seed(42)
        a = np.random.randn(100) + 0.5
        b = np.random.randn(100) - 0.5
        result = paired_significance_test(a, b)
        assert result["significant"] is True

    def test_short_series(self) -> None:
        """过短序列应返回insufficient_data。"""
        a = np.array([0.01])
        b = np.array([0.02])
        result = paired_significance_test(a, b)
        assert result["method"] == "insufficient_data"


class TestExperiment4OutputFiles:
    """实验4输出文件完整性测试。"""

    EXP_DIR = Path(__file__).resolve().parent.parent / "data" / "experiment4"

    def test_summary_json_exists(self) -> None:
        """experiment4_summary.json应存在。"""
        path = self.EXP_DIR / "experiment4_summary.json"
        assert path.exists(), f"缺少文件: {path}"
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert "generated_at" in data
        assert "methods" in data
        assert len(data["methods"]) >= 2

    def test_report_md_exists(self) -> None:
        """experiment4_report.md应存在。"""
        path = self.EXP_DIR / "experiment4_report.md"
        assert path.exists(), f"缺少文件: {path}"
        content = path.read_text(encoding="utf-8")
        assert "综合对比" in content
        assert "分年度分析" in content
        assert "多空组合" in content
        assert "统计检验" in content

    def test_yearly_analysis_json_exists(self) -> None:
        """yearly_analysis.json应存在且包含年度数据。"""
        path = self.EXP_DIR / "yearly_analysis.json"
        assert path.exists(), f"缺少文件: {path}"
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) >= 2
        for name, yearly in data.items():
            assert len(yearly) >= 1
            for year, metrics in yearly.items():
                assert "annualized_return" in metrics
                assert "sharpe_ratio" in metrics

    def test_quantile_analysis_json_exists(self) -> None:
        """quantile_analysis.json应存在且包含分位数数据。"""
        path = self.EXP_DIR / "quantile_analysis.json"
        assert path.exists(), f"缺少文件: {path}"
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) >= 2
        for name, qd in data.items():
            assert "Q1" in qd
            assert "Q5" in qd
            assert "LS" in qd

    def test_significance_tests_json_exists(self) -> None:
        """significance_tests.json应存在。"""
        path = self.EXP_DIR / "significance_tests.json"
        assert path.exists(), f"缺少文件: {path}"
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, list)
        if len(data) > 0:
            assert "method" in data[0]
            assert "paired_test" in data[0]

    def test_backtest_result_parquets_exist(self) -> None:
        """各方法的回测结果parquet文件应存在。"""
        for name in ["M1-B-LGBM", "M2-B-XGB"]:
            path = self.EXP_DIR / f"{name}_backtest_result.parquet"
            assert path.exists(), f"缺少文件: {path}"

    def test_backtest_metrics_jsons_exist(self) -> None:
        """各方法的回测指标JSON文件应存在。"""
        for name in ["M1-B-LGBM", "M2-B-XGB"]:
            path = self.EXP_DIR / f"{name}_backtest_metrics.json"
            assert path.exists(), f"缺少文件: {path}"
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            assert "annualized_return" in data
            assert "sharpe_ratio" in data


class TestSafeFloat:
    """_safe_float 辅助函数测试。"""

    def test_normal_value(self) -> None:
        """正常浮点值应原样返回。"""
        from scripts.run_experiment4 import _safe_float
        assert _safe_float(3.14) == 3.14

    def test_none_value(self) -> None:
        """None应返回0.0。"""
        from scripts.run_experiment4 import _safe_float
        assert _safe_float(None) == 0.0

    def test_nan_value(self) -> None:
        """NaN应返回0.0。"""
        from scripts.run_experiment4 import _safe_float
        assert _safe_float(float("nan")) == 0.0

    def test_inf_value(self) -> None:
        """Inf应返回0.0。"""
        from scripts.run_experiment4 import _safe_float
        assert _safe_float(float("inf")) == 0.0

    def test_string_value(self) -> None:
        """无法转换的字符串应返回0.0。"""
        from scripts.run_experiment4 import _safe_float
        assert _safe_float("abc") == 0.0


class TestGetMethodConfigurations:
    """方法配置测试。"""

    def test_four_methods_defined(self) -> None:
        """应定义4种方法。"""
        from scripts.run_experiment4 import get_method_configurations
        configs = get_method_configurations()
        assert len(configs) == 4
        assert "M1-B-LGBM" in configs
        assert "M2-B-XGB" in configs

    def test_each_method_has_required_keys(self) -> None:
        """每种方法配置应包含必要键。"""
        from scripts.run_experiment4 import get_method_configurations
        configs = get_method_configurations()
        for name, config in configs.items():
            assert "label" in config
            assert "predictor_type" in config
            assert "vol_penalty" in config
