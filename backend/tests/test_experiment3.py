"""
实验3相关模块的单元测试：覆盖 feature_contribution.py 和 run_experiment3.py 的核心函数。

测试范围：
- compute_feature_contribution: 正常流程、空数据、缺失列、NaN处理
- build_e3a_prompt / build_e3b_prompt: 提示词生成格式
- parse_llm_stock_suggestions: LLM回复解析
- calculate_consistency: 一致率计算
- find_extreme_sections: 截面选择逻辑
- generate_case_report / generate_experiment_report: 报告生成
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.feature_contribution import compute_feature_contribution


@pytest.fixture
def sample_panel_df() -> pd.DataFrame:
    """构造含3只股票、4个因子的截面数据。"""
    rng = np.random.default_rng(42)
    n = 3
    return pd.DataFrame({
        "stock_code": ["000001", "000002", "000003"],
        "factor_a": rng.standard_normal(n),
        "factor_b": rng.standard_normal(n),
        "factor_c": rng.standard_normal(n),
        "factor_d": rng.standard_normal(n),
    })


@pytest.fixture
def sample_lgb_model(sample_panel_df: pd.DataFrame) -> lgb.Booster:
    """使用小数据集训练一个最小LightGBM模型，用于predict_contrib测试。"""
    rng = np.random.default_rng(42)
    n = 50
    X = rng.standard_normal((n, 4))
    y_float = X[:, 0] * 2 + X[:, 1] - X[:, 2]
    y_int = np.clip(np.round(y_float * 5 + 15), 0, 30).astype(np.int32)
    groups = [10] * 5

    train_data = lgb.Dataset(X, label=y_int, group=groups, free_raw_data=False)
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "verbose": -1,
        "num_leaves": 4,
        "min_child_samples": 2,
        "seed": 42,
    }
    model = lgb.train(params, train_data, num_boost_round=10)
    return model


class TestComputeFeatureContribution:
    """compute_feature_contribution 函数测试。"""

    def test_normal_flow(
        self,
        sample_lgb_model: lgb.Booster,
        sample_panel_df: pd.DataFrame,
    ) -> None:
        """正常流程：返回字典包含所有stock_code，每只股票有top_k个因子。"""
        factor_cols = ["factor_a", "factor_b", "factor_c", "factor_d"]
        result = compute_feature_contribution(
            sample_lgb_model, sample_panel_df, factor_cols, top_k=3,
        )

        assert isinstance(result, dict)
        assert len(result) == 3
        for code in ["000001", "000002", "000003"]:
            assert code in result
            entries = result[code]
            assert len(entries) == 3
            for fname, cval in entries:
                assert isinstance(fname, str)
                assert isinstance(cval, float)

    def test_top_k_ordering(
        self,
        sample_lgb_model: lgb.Booster,
        sample_panel_df: pd.DataFrame,
    ) -> None:
        """贡献值按绝对值降序排列。"""
        factor_cols = ["factor_a", "factor_b", "factor_c", "factor_d"]
        result = compute_feature_contribution(
            sample_lgb_model, sample_panel_df, factor_cols, top_k=4,
        )

        for code, entries in result.items():
            abs_vals = [abs(v) for _, v in entries]
            assert abs_vals == sorted(abs_vals, reverse=True), (
                f"stock {code} 的贡献值未按绝对值降序排列"
            )

    def test_empty_dataframe_raises(self, sample_lgb_model: lgb.Booster) -> None:
        """空DataFrame应抛出ValueError。"""
        empty_df = pd.DataFrame(columns=["stock_code", "factor_a"])
        with pytest.raises(ValueError, match="panel_df 为空"):
            compute_feature_contribution(sample_lgb_model, empty_df, ["factor_a"])

    def test_missing_stock_code_raises(
        self,
        sample_lgb_model: lgb.Booster,
    ) -> None:
        """缺少stock_code列应抛出ValueError。"""
        df = pd.DataFrame({"factor_a": [1.0, 2.0]})
        with pytest.raises(ValueError, match="stock_code"):
            compute_feature_contribution(sample_lgb_model, df, ["factor_a"])

    def test_missing_factor_cols_raises(
        self,
        sample_lgb_model: lgb.Booster,
    ) -> None:
        """缺少因子列应抛出ValueError。"""
        df = pd.DataFrame({"stock_code": ["000001"], "factor_a": [1.0]})
        with pytest.raises(ValueError, match="缺少因子列"):
            compute_feature_contribution(sample_lgb_model, df, ["factor_a", "missing_col"])

    def test_nan_fill(
        self,
        sample_lgb_model: lgb.Booster,
    ) -> None:
        """含NaN值的因子列应被中位数填充，不抛出异常。"""
        df = pd.DataFrame({
            "stock_code": ["000001", "000002"],
            "factor_a": [1.0, np.nan],
            "factor_b": [np.nan, 2.0],
            "factor_c": [3.0, 4.0],
            "factor_d": [5.0, 6.0],
        })
        factor_cols = ["factor_a", "factor_b", "factor_c", "factor_d"]
        result = compute_feature_contribution(sample_lgb_model, df, factor_cols, top_k=2)
        assert len(result) == 2

    def test_top_k_less_than_factors(
        self,
        sample_lgb_model: lgb.Booster,
        sample_panel_df: pd.DataFrame,
    ) -> None:
        """top_k小于因子数量时，每只股票只返回top_k个。"""
        factor_cols = ["factor_a", "factor_b", "factor_c", "factor_d"]
        result = compute_feature_contribution(
            sample_lgb_model, sample_panel_df, factor_cols, top_k=2,
        )
        for entries in result.values():
            assert len(entries) == 2


class TestBuildE3aPrompt:
    """build_e3a_prompt 函数测试。"""

    def test_contains_key_sections(self) -> None:
        """E3a提示词应包含关键章节标题和表格。"""
        from scripts.run_experiment3 import build_e3a_prompt

        top_stocks = [
            {"stock_code": "000001", "score": 0.9},
            {"stock_code": "600000", "score": 0.8},
        ]
        fi = {"factor_a": 100.0, "factor_b": 50.0}

        text = build_e3a_prompt("2024-01-01", top_stocks, fi)

        assert "模型预测结果" in text
        assert "2024-01-01" in text
        assert "Top-K推荐股票" in text
        assert "000001" in text
        assert "0.9000" in text
        assert "全局重要因子" in text
        assert "factor_a" in text

    def test_empty_top_stocks(self) -> None:
        """空股票列表应生成无数据的表格。"""
        from scripts.run_experiment3 import build_e3a_prompt

        text = build_e3a_prompt("2024-01-01", [], {"f1": 10.0})
        assert "模型预测结果" in text


class TestBuildE3bPrompt:
    """build_e3b_prompt 函数测试。"""

    def test_contains_contribution_section(self) -> None:
        """E3b提示词应包含特征贡献分解章节。"""
        from scripts.run_experiment3 import build_e3b_prompt

        e3a_text = "## 模型预测结果\n### Top-K推荐股票"
        top_stocks = [{"stock_code": "000001", "score": 0.9}]
        contributions = {
            "000001": [("factor_a", 0.5), ("factor_b", -0.3)],
        }

        text = build_e3b_prompt(e3a_text, top_stocks, contributions)

        assert "特征贡献分解" in text
        assert "000001" in text
        assert "factor_a" in text
        assert "正向" in text
        assert "负向" in text

    def test_direction_labels(self) -> None:
        """正贡献标记为正向，负贡献标记为负向。"""
        from scripts.run_experiment3 import build_e3b_prompt

        e3a_text = ""
        top_stocks = [{"stock_code": "000001", "score": 0.5}]
        contributions = {
            "000001": [("f_pos", 0.1), ("f_neg", -0.2)],
        }

        text = build_e3b_prompt(e3a_text, top_stocks, contributions)
        lines = text.split("\n")

        pos_line = [l for l in lines if "f_pos" in l]
        neg_line = [l for l in lines if "f_neg" in l]
        assert any("正向" in l for l in pos_line)
        assert any("负向" in l for l in neg_line)


class TestParseLLMStockSuggestions:
    """parse_llm_stock_suggestions 函数测试。"""

    def test_parse_sz_sh_codes(self) -> None:
        """应正确解析带.SZ/.SH后缀的股票代码。"""
        from scripts.run_experiment3 import parse_llm_stock_suggestions

        text = "建议关注 000001.SZ 和 600000.SH"
        codes = parse_llm_stock_suggestions(text)
        assert "000001.SZ" in codes
        assert "600000.SH" in codes

    def test_parse_with_prefix(self) -> None:
        """应解析带买入/卖出前缀的代码。"""
        from scripts.run_experiment3 import parse_llm_stock_suggestions

        text = "买入：000001\n卖出：600036"
        codes = parse_llm_stock_suggestions(text)
        assert "000001" in codes
        assert "600036" in codes

    def test_empty_text(self) -> None:
        """空文本应返回空集合。"""
        from scripts.run_experiment3 import parse_llm_stock_suggestions

        codes = parse_llm_stock_suggestions("")
        assert len(codes) == 0

    def test_no_stock_codes(self) -> None:
        """无股票代码的文本应返回空集合。"""
        from scripts.run_experiment3 import parse_llm_stock_suggestions

        codes = parse_llm_stock_suggestions("这是一段普通文本，没有股票代码。")
        assert len(codes) == 0


class TestCalculateConsistency:
    """calculate_consistency 函数测试。"""

    def test_full_match(self) -> None:
        """完全匹配时一致率为1。"""
        from scripts.run_experiment3 import calculate_consistency

        llm_codes = {"000001", "000002", "000003"}
        model_top = ["000001", "000002", "000003"]
        result = calculate_consistency(llm_codes, model_top)
        assert result["matched_count"] == 3
        assert result["consistency_rate"] == 1.0

    def test_partial_match(self) -> None:
        """部分匹配时一致率正确。"""
        from scripts.run_experiment3 import calculate_consistency

        llm_codes = {"000001", "000002", "999999"}
        model_top = ["000001", "000002", "000003"]
        result = calculate_consistency(llm_codes, model_top)
        assert result["matched_count"] == 2
        assert result["consistency_rate"] == pytest.approx(2 / 3, abs=0.01)

    def test_no_match(self) -> None:
        """无匹配时一致率为0。"""
        from scripts.run_experiment3 import calculate_consistency

        llm_codes = {"111111", "222222"}
        model_top = ["000001", "000002"]
        result = calculate_consistency(llm_codes, model_top)
        assert result["matched_count"] == 0
        assert result["consistency_rate"] == 0.0

    def test_empty_llm_codes(self) -> None:
        """LLM建议为空时一致率为0。"""
        from scripts.run_experiment3 import calculate_consistency

        result = calculate_consistency(set(), ["000001"])
        assert result["consistency_rate"] == 0.0

    def test_empty_model_top(self) -> None:
        """模型Top-K为空时一致率为0。"""
        from scripts.run_experiment3 import calculate_consistency

        result = calculate_consistency({"000001"}, [])
        assert result["consistency_rate"] == 0.0


class TestFindExtremeSections:
    """find_extreme_sections 函数测试。"""

    def test_selects_bull_and_bear(self) -> None:
        """应正确选择收益率最高和最低的截面。"""
        from scripts.run_experiment3 import find_extreme_sections

        rng = np.random.default_rng(42)
        dates = pd.date_range("2023-01-01", periods=4, freq="W-FRI")
        rows = []
        returns_by_date = [0.05, -0.10, 0.02, -0.03]
        for i, dt in enumerate(dates):
            for j in range(200):
                rows.append({
                    "date": dt,
                    "stock_code": f"{j:06d}",
                    "factor_a": rng.standard_normal(),
                    "future_return_1w": returns_by_date[i] + rng.standard_normal() * 0.01,
                })
        df = pd.DataFrame(rows)

        bull_df, bull_date, bull_ret, bear_df, bear_date, bear_ret = find_extreme_sections(df)

        assert bull_date == "2023-01-06"
        assert bear_date == "2023-01-13"
        assert bull_ret > bear_ret

    def test_insufficient_data_raises(self) -> None:
        """每个日期股票数不足100时应抛出异常。"""
        from scripts.run_experiment3 import find_extreme_sections

        df = pd.DataFrame({
            "date": ["2023-01-06"] * 10,
            "stock_code": [f"{i:06d}" for i in range(10)],
            "future_return_1w": np.random.randn(10),
        })
        with pytest.raises(ValueError, match="无法找到有效截面"):
            find_extreme_sections(df)


class TestGenerateCaseReport:
    """generate_case_report 函数测试。"""

    def test_report_structure(self) -> None:
        """报告应包含E3a、E3b、E3c三个章节。"""
        from scripts.run_experiment3 import generate_case_report

        report = generate_case_report(
            "测试案例", "2024-01-01", 0.05,
            "E3a内容", "E3b内容", "", None, False,
        )
        assert "# 测试案例" in report
        assert "E3a" in report
        assert "E3b" in report
        assert "E3c" in report
        assert "2024-01-01" in report

    def test_e3c_success_report(self) -> None:
        """E3c成功时报告应包含一致性评估。"""
        from scripts.run_experiment3 import generate_case_report

        consistency = {
            "llm_suggestion_count": 5,
            "model_top_k_count": 10,
            "matched_count": 3,
            "matched_codes": ["000001", "000002", "000003"],
            "consistency_rate": 0.6,
        }
        report = generate_case_report(
            "测试案例", "2024-01-01", 0.05,
            "E3a内容", "E3b内容", "LLM回复内容", consistency, True,
        )
        assert "一致性评估" in report
        assert "60.00%" in report

    def test_e3c_failure_report(self) -> None:
        """E3c失败时报告应显示警告信息。"""
        from scripts.run_experiment3 import generate_case_report

        report = generate_case_report(
            "测试案例", "2024-01-01", -0.05,
            "E3a内容", "E3b内容", "", None, False,
        )
        assert "E3c未生成" in report


class TestGenerateExperimentReport:
    """generate_experiment_report 函数测试。"""

    def test_report_contains_all_sections(self) -> None:
        """总报告应包含实验设计、截面选择、执行结果等章节。"""
        from scripts.run_experiment3 import generate_experiment_report

        report = generate_experiment_report(
            case1_e3c_success=True,
            case2_e3c_success=False,
            case1_consistency={"consistency_rate": 0.5, "llm_suggestion_count": 5, "matched_count": 2, "matched_codes": [], "model_top_k_count": 10},
            case2_consistency=None,
        )
        assert "实验设计" in report
        assert "截面选择" in report
        assert "执行结果" in report
        assert "关键结论" in report
        assert "文件清单" in report

    def test_both_e3c_success(self) -> None:
        """两个案例都成功时应显示平均一致率。"""
        from scripts.run_experiment3 import generate_experiment_report

        cons1 = {"consistency_rate": 0.6, "llm_suggestion_count": 5, "matched_count": 3, "matched_codes": [], "model_top_k_count": 10}
        cons2 = {"consistency_rate": 0.4, "llm_suggestion_count": 5, "matched_count": 2, "matched_codes": [], "model_top_k_count": 10}

        report = generate_experiment_report(True, True, cons1, cons2)
        assert "平均一致率" in report
        assert "50.00%" in report


class TestCallDeepseekForE3c:
    """call_deepseek_for_e3c 函数测试（Mock DeepSeek API）。"""

    def test_api_key_missing_returns_false(self) -> None:
        """API Key缺失时应返回空字符串和False。"""
        from scripts.run_experiment3 import call_deepseek_for_e3c

        with patch("src.deepseek_stream._api_key", return_value=""):
            text, success = call_deepseek_for_e3c("test", [])
            assert text == ""
            assert success is False

    def test_api_failure_returns_false(self) -> None:
        """API调用失败时应返回空字符串和False。"""
        from scripts.run_experiment3 import call_deepseek_for_e3c

        with patch("src.deepseek_stream._api_key", return_value="fake-key"), \
             patch("src.deepseek_stream.get_advice_sync_blocking", side_effect=Exception("API error")):
            text, success = call_deepseek_for_e3c("test", [])
            assert text == ""
            assert success is False

    def test_api_success_returns_text(self) -> None:
        """API调用成功时应返回回复文本和True。"""
        from scripts.run_experiment3 import call_deepseek_for_e3c

        mock_response = "## 操作建议\n建议买入 000001.SZ"
        with patch("src.deepseek_stream._api_key", return_value="fake-key"), \
             patch("src.deepseek_stream.get_advice_sync_blocking", return_value=mock_response), \
             patch("src.prompt_builder.build_system_prompt", return_value="system"), \
             patch("src.prompt_builder.compose_user_prompt_text", return_value="user"):
            text, success = call_deepseek_for_e3c("test", [{"stock_code": "000001", "score": 0.9}])
            assert success is True
            assert "操作建议" in text


class TestGenerateTemplateE3c:
    """generate_template_e3c 函数测试。"""

    def test_bull_market_template(self) -> None:
        """牛市截面模板应包含买入建议和偏多环境描述。"""
        from scripts.run_experiment3 import generate_template_e3c

        top_stocks = [
            {"stock_code": "000001", "score": 0.9},
            {"stock_code": "600000", "score": 0.8},
        ]
        contributions = {
            "000001": [("factor_a", 0.5), ("factor_b", -0.3)],
            "600000": [("factor_c", 0.4)],
        }
        fi = {"factor_a": 100.0, "factor_b": 50.0}
        text = generate_template_e3c("2024-09-20", top_stocks, contributions, fi, 0.1388)

        assert "操作建议" in text
        assert "买入建议" in text
        assert "卖出建议" in text
        assert "调仓与持仓调整" in text
        assert "风险提示" in text
        assert "偏多" in text
        assert "000001" in text
        assert "600000" in text

    def test_bear_market_template(self) -> None:
        """熊市截面模板应包含偏空环境和降仓建议。"""
        from scripts.run_experiment3 import generate_template_e3c

        top_stocks = [{"stock_code": "000001", "score": 0.5}]
        contributions = {"000001": [("factor_a", 0.2)]}
        fi = {"factor_a": 100.0}
        text = generate_template_e3c("2024-01-26", top_stocks, contributions, fi, -0.1416)

        assert "偏空" in text
        assert "降低整体仓位" in text

    def test_template_contains_stock_codes(self) -> None:
        """模板应包含可被parse_llm_stock_suggestions解析的股票代码格式。"""
        from scripts.run_experiment3 import generate_template_e3c, parse_llm_stock_suggestions

        top_stocks = [
            {"stock_code": "000001", "score": 0.9},
            {"stock_code": "600000", "score": 0.8},
            {"stock_code": "300001", "score": 0.7},
        ]
        contributions = {
            "000001": [("f1", 0.5)],
            "600000": [("f2", 0.4)],
            "300001": [("f3", 0.3)],
        }
        fi = {"f1": 100.0}
        text = generate_template_e3c("2024-01-01", top_stocks, contributions, fi, 0.05)

        codes = parse_llm_stock_suggestions(text)
        assert "000001" in codes
        assert "600000" in codes
        assert "300001" in codes

    def test_empty_stocks(self) -> None:
        """空股票列表应不崩溃。"""
        from scripts.run_experiment3 import generate_template_e3c

        text = generate_template_e3c("2024-01-01", [], {}, {}, 0.01)
        assert "操作建议" in text
