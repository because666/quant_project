"""
实验3：AI推荐信息粒度对比

选取测试集中周收益率最高（牛市截面）和最低（熊市截面）的两个典型截面，
生成三种信息粒度下的推荐案例：
- E3a: 模型输出 + Top因子名称（基础粒度）
- E3b: 在E3a基础上增加特征贡献分解（中等粒度）
- E3c: 在E3b基础上调用DeepSeek LLM生成完整推荐（高粒度）

用法：
    cd backend
    python scripts/run_experiment3.py
"""
from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime, date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_settings
from src.data_loader import DATA_OUT_DIR, load_factor_columns, fill_missing_factors
from src.feature_contribution import compute_feature_contribution
from src.model_lightgbm import load_lightgbm_model
from src.predictor import ModelPredictor
from src.prompt_builder import build_system_prompt, compose_user_prompt_text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
EXPERIMENT_DIR = DATA_DIR / "experiment3"

TOP_K_STOCKS: int = 10
TOP_K_FACTORS: int = 5
CONTRIB_TOP_K: int = 5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("experiment3")


def load_test_data() -> pd.DataFrame:
    """
    加载测试集数据并填充缺失值。

    返回:
        清洗后的测试集DataFrame，含date、stock_code、因子列和future_return_1w列
    """
    path = DATA_DIR / "test.parquet"
    if not path.exists():
        raise FileNotFoundError(f"测试集文件不存在: {path}")

    df = pd.read_parquet(path)
    factor_cols = load_factor_columns(data_dir=DATA_DIR)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["future_return_1w"] = pd.to_numeric(df["future_return_1w"], errors="coerce")

    for col in factor_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = fill_missing_factors(df, factor_cols=factor_cols, method="median")
    logger.info("测试集加载完成: %d行, %d列, 日期范围 %s ~ %s", len(df), len(df.columns), df["date"].min(), df["date"].max())
    return df


def find_extreme_sections(
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, str, str, pd.DataFrame, str, str]:
    """
    从测试集中找出周收益率最高的一周（牛市截面）和最低的一周（熊市截面）。

    参数:
        test_df: 测试集数据，需包含 date 和 future_return_1w 列

    返回:
        (bull_section_df, bull_date_str, bull_avg_return,
         bear_section_df, bear_date_str, bear_avg_return)
    """
    section_stats = (
        test_df.groupby("date")["future_return_1w"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "avg_return"})
    )
    section_stats = section_stats[section_stats["count"] >= 100]

    if section_stats.empty:
        raise ValueError("无法找到有效截面（每个日期至少需要100只股票）")

    bull_idx = section_stats["avg_return"].idxmax()
    bear_idx = section_stats["avg_return"].idxmin()

    bull_date = section_stats.loc[bull_idx, "date"]
    bear_date = section_stats.loc[bear_idx, "date"]

    bull_return = float(section_stats.loc[bull_idx, "avg_return"])
    bear_return = float(section_stats.loc[bear_idx, "avg_return"])

    bull_date_str = pd.Timestamp(bull_date).strftime("%Y-%m-%d")
    bear_date_str = pd.Timestamp(bear_date).strftime("%Y-%m-%d")

    bull_df = test_df[test_df["date"] == bull_date].copy()
    bear_df = test_df[test_df["date"] == bear_date].copy()

    logger.info(
        "牛市截面: %s, 平均收益率=%.4f%%, 股票数=%d",
        bull_date_str, bull_return * 100, len(bull_df),
    )
    logger.info(
        "熊市截面: %s, 平均收益率=%.4f%%, 股票数=%d",
        bear_date_str, bear_return * 100, len(bear_df),
    )

    return bull_df, bull_date_str, bull_return, bear_df, bear_date_str, bear_return


def build_e3a_prompt(
    date_str: str,
    top_stocks: list[dict[str, Any]],
    feature_importance: dict[str, float],
) -> str:
    """
    构造E3a提示词：模型预测结果 + Top-K股票 + 全局重要因子。

    参数:
        date_str: 截面日期字符串，格式 YYYY-MM-DD
        top_stocks: Top-K股票列表，每项包含 stock_code 和 score 字段
        feature_importance: 全局特征重要性字典 {factor_name: importance_value}

    返回:
        Markdown格式的E3a提示词文本
    """
    lines: list[str] = []
    lines.append(f"## 模型预测结果（{date_str}）")
    lines.append("")
    lines.append("### Top-K推荐股票")
    lines.append("")
    lines.append("| 排名 | 股票代码 | 模型得分 |")
    lines.append("|------|---------|---------|")
    for i, stock in enumerate(top_stocks[:TOP_K_STOCKS], start=1):
        code = stock.get("stock_code", "")
        score = stock.get("score", 0.0)
        lines.append(f"| {i} | {code} | {score:.4f} |")
    lines.append("")

    sorted_fi = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:TOP_K_FACTORS]
    lines.append("### 全局重要因子（Top 5）")
    lines.append("")
    lines.append("| 因子名 | 重要性 |")
    lines.append("|--------|-------|")
    for name, imp in sorted_fi:
        lines.append(f"| {name} | {imp:.6g} |")
    lines.append("")

    return "\n".join(lines)


def build_e3b_prompt(
    e3a_text: str,
    top_stocks: list[dict[str, Any]],
    contributions: dict[str, list[tuple[str, float]]],
) -> str:
    """
    构造E3b提示词：在E3a基础上增加每只Top-K股票的特征贡献分解。

    参数:
        e3a_text: E3a提示词文本
        top_stocks: Top-K股票列表
        contributions: 特征贡献字典 {stock_code: [(factor_name, contribution), ...]}

    返回:
        Markdown格式的E3b提示词文本
    """
    lines: list[str] = [e3a_text]
    lines.append("### Top-K股票特征贡献分解")
    lines.append("")
    lines.append("以下为各推荐股票的Top-K因子贡献值（SHAP近似），正值表示该因子推高得分，负值表示拉低得分。")
    lines.append("")

    for stock in top_stocks[:TOP_K_STOCKS]:
        code = stock.get("stock_code", "")
        contrib_list = contributions.get(code, [])
        lines.append(f"**{code} 的特征贡献分解**")
        lines.append("")
        lines.append("| 因子名 | 贡献值 | 方向 |")
        lines.append("|--------|-------|------|")
        for fname, cval in contrib_list[:CONTRIB_TOP_K]:
            direction = "正向" if cval > 0 else "负向"
            lines.append(f"| {fname} | {cval:.6f} | {direction} |")
        lines.append("")

    return "\n".join(lines)


def generate_template_e3c(
    date_str: str,
    top_stocks: list[dict[str, Any]],
    contributions: dict[str, list[tuple[str, float]]],
    feature_importance: dict[str, float],
    avg_return: float,
) -> str:
    """
    当DeepSeek API不可用时，基于E3b数据生成模板化的E3c推荐。

    遵循与LLM相同的输出格式（Markdown结构），确保一致性评估可执行。

    参数:
        date_str: 截面日期字符串
        top_stocks: Top-K股票列表
        contributions: 特征贡献字典
        feature_importance: 全局特征重要性字典
        avg_return: 截面平均收益率

    返回:
        Markdown格式的模板化推荐文本
    """
    market_regime = "偏多" if avg_return > 0 else "偏空"
    risk_level = "高" if abs(avg_return) > 0.05 else "中"

    lines: list[str] = []
    lines.append("## 操作建议")
    lines.append("")
    lines.append("### 买入建议")
    lines.append("")

    top5 = top_stocks[:5]
    for i, stock in enumerate(top5, start=1):
        code = stock.get("stock_code", "")
        score = stock.get("score", 0.0)
        contrib_list = contributions.get(code, [])
        top_factors = [f"{fname}({cval:+.4f})" for fname, cval in contrib_list[:3]]
        factor_desc = "、".join(top_factors) if top_factors else "无显著因子"
        lines.append(
            f"{i}. 关注 {code}（模型得分 {score:.4f}）："
            f"模型排名靠前，主要受{factor_desc}驱动。"
            f"建议关注，可分批建仓。"
        )

    lines.append("")
    lines.append("### 卖出建议")
    lines.append("")
    lines.append(
        f"当前市场环境{market_regime}，截面平均周收益率{avg_return * 100:.2f}%。"
        f"对于不在模型Top-10推荐中的持仓，建议评估是否减仓或止损。"
    )
    lines.append("")

    bottom_stocks = top_stocks[5:]
    if bottom_stocks:
        lines.append("Top-10中排名靠后的标的（第6-10名）：")
        for stock in bottom_stocks:
            code = stock.get("stock_code", "")
            score = stock.get("score", 0.0)
            lines.append(f"- 持有 {code}（得分 {score:.4f}）：得分相对较低，建议谨慎持有，设置止损。")
        lines.append("")

    lines.append("### 调仓与持仓调整")
    lines.append("")
    lines.append(
        f"当前截面市场风险等级{risk_level}，建议："
    )
    lines.append(
        "- 优先配置模型Top-5标的，等权或按得分加权；"
    )
    lines.append(
        "- 控制单只标的仓位不超过组合的15%；"
    )
    if market_regime == "偏空":
        lines.append(
            "- 市场偏空环境下，建议降低整体仓位至50%以下，保留现金观望；"
        )
    else:
        lines.append(
            "- 市场偏多环境下，可适度提高仓位至80%，但仍需设置止损线；"
        )
    lines.append(
        "- 关注模型重要因子（boll_width、realized_var_8w、ma_ratio_8w）的变化趋势。"
    )
    lines.append("")

    lines.append("## 风险提示")
    lines.append("")
    lines.append("- 本建议基于排序学习模型的历史统计规律，不构成投资建议；")
    lines.append("- 模型可能存在过拟合风险，历史表现不代表未来收益；")
    lines.append(
        f"- 当前截面波动率{risk_level}，需警惕市场风格切换导致的模型失效；"
    )
    lines.append("- 建议结合基本面和宏观环境综合判断，避免单一依赖模型信号。")
    lines.append("")

    return "\n".join(lines)


def call_deepseek_for_e3c(
    e3b_text: str,
    top_stocks: list[dict[str, Any]],
) -> tuple[str, bool]:
    """
    调用DeepSeek LLM生成E3c完整推荐。

    使用已有的prompt_builder构造系统/用户消息，
    通过deepseek_stream获取同步响应。

    参数:
        e3b_text: E3b提示词文本（作为用户消息的补充上下文）
        top_stocks: Top-K股票列表（用于构造标准prompt）

    返回:
        (LLM回复文本, 是否成功调用)

    异常:
        无异常抛出；失败时返回空字符串和False
    """
    try:
        from src.deepseek_stream import get_advice_sync_blocking, _api_key
    except ImportError:
        logger.warning("deepseek_stream 模块导入失败，跳过E3c")
        return "", False

    try:
        api_key = _api_key()
        if not api_key or not api_key.strip():
            logger.warning("DEEPSEEK_API_KEY 未配置或为空，跳过E3c")
            return "", False
    except Exception:
        logger.warning("无法读取DEEPSEEK_API_KEY，跳过E3c")
        return "", False

    system_msg = build_system_prompt()
    user_top_stocks = [
        {"stock_code": s.get("stock_code", ""), "score": s.get("score", 0.0)}
        for s in top_stocks[:TOP_K_STOCKS]
    ]

    base_user_text = compose_user_prompt_text(
        top_stocks=user_top_stocks,
        holdings=None,
        price_ranges=None,
        feature_importance=None,
    )

    full_user_text = f"{base_user_text}\n\n---\n\n以下是当前截面的详细模型分析数据，请结合这些信息给出更精准的调仓建议：\n\n{e3b_text}"

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": full_user_text},
    ]

    try:
        logger.info("正在调用DeepSeek API生成E3c推荐...")
        response = get_advice_sync_blocking(messages)
        logger.info("DeepSeek API调用成功，回复长度=%d字符", len(response))
        return response, True
    except Exception as exc:
        logger.error("DeepSeek API调用失败: %s，跳过E3c", exc)
        return "", False


def parse_llm_stock_suggestions(response_text: str) -> set[str]:
    """
    从LLM回复中解析买入/卖出建议的股票代码。

    匹配规则：
    - A股6位数字代码（000001~999999）
    - 可能带.SZ/.SH后缀的代码
    - 支持"关注/推荐/持有/买入"等关键词后跟代码的格式

    参数:
        response_text: LLM回复的Markdown文本

    返回:
        解析到的股票代码集合
    """
    patterns = [
        r'(\d{6}\.(?:SZ|SH))',
        r'(?:买入|关注|推荐|持有|加仓)\s*[：:]*\s*(\d{6}(?:\.(?:SZ|SH))?)',
        r'(?:卖出|减仓|止损|止盈|清仓)\s*[：:]*\s*(\d{6}(?:\.(?:SZ|SH))?)',
        r'\b([036]\d{5})\b(?=\s*(?:股|号|代码|买入|卖出|关注|持有))',
        r'(?:关注|推荐|持有|买入)\s+([036]\d{5})\b',
        r'\*{1,2}([036]\d{5})\*{1,2}',
        r'`([036]\d{5})`',
        r'\b([036]\d{5})\b',
    ]
    codes: set[str] = set()
    for pattern in patterns:
        matches = re.findall(pattern, response_text)
        for m in matches:
            code = m if isinstance(m, str) else m[0] if isinstance(m, tuple) else ""
            if code and len(str(code)) >= 4:
                codes.add(str(code))
    return codes


def calculate_consistency(
    llm_codes: set[str],
    model_top_codes: list[str],
) -> dict[str, Any]:
    """
    计算LLM建议与模型Top-K的一致率。

    一致率 = 匹配数 / min(LLM建议数, Top-K数量)

    参数:
        llm_codes: LLM建议的股票代码集合
        model_top_codes: 模型Top-K股票代码列表

    返回:
        包含一致率、匹配数等信息的字典
    """
    model_set = set(model_top_codes)
    matched = llm_codes & model_set
    denominator = min(len(llm_codes), len(model_top_codes)) if llm_codes and model_top_codes else 1
    consistency_rate = len(matched) / denominator if denominator > 0 else 0.0

    return {
        "llm_suggestion_count": len(llm_codes),
        "model_top_k_count": len(model_top_codes),
        "matched_count": len(matched),
        "matched_codes": sorted(matched),
        "consistency_rate": round(consistency_rate, 4),
    }


def generate_case_report(
    case_name: str,
    date_str: str,
    avg_return: float,
    e3a_text: str,
    e3b_text: str,
    e3c_text: str,
    consistency: dict[str, Any] | None,
    e3c_success: bool,
    e3c_is_template: bool = False,
) -> str:
    """
    生成单个案例的完整报告（Markdown格式）。

    参数:
        case_name: 案例名称（如 "牛市截面案例"）
        date_str: 截面日期
        avg_return: 该截面平均周收益率
        e3a_text: E3a提示词
        e3b_text: E3b提示词
        e3c_text: E3c LLM回复
        consistency: 一致性评估结果
        e3c_success: E3c是否成功调用
        e3c_is_template: E3c是否为模板生成（非LLM）

    返回:
        Markdown格式的案例报告文本
    """
    lines: list[str] = []
    lines.append(f"# {case_name}")
    lines.append("")
    lines.append(f"- **截面日期**: {date_str}")
    lines.append(f"- **截面平均周收益率**: {avg_return * 100:.4f}%")
    lines.append(f"- **报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## E3a：基础粒度（模型输出 + Top因子名称）")
    lines.append("")
    lines.append(e3a_text)
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## E3b：中等粒度（增加特征贡献分解）")
    lines.append("")
    lines.append(e3b_text)
    lines.append("")

    lines.append("---")
    lines.append("")
    if e3c_success and e3c_text:
        source_label = "模板生成" if e3c_is_template else "DeepSeek LLM"
        lines.append(f"## E3c：高粒度（{source_label}完整推荐）")
        if e3c_is_template:
            lines.append("")
            lines.append("> ⚠️ **注意**：DeepSeek API不可用，以下E3c由模板规则生成，非LLM输出。API恢复后请重新运行以获取真实LLM推荐。")
        lines.append("")
        lines.append(e3c_text)
        lines.append("")

        if consistency is not None:
            lines.append("---")
            lines.append("")
            lines.append("### E3c 一致性评估")
            lines.append("")
            lines.append("- **LLM建议股票数**: {}".format(consistency["llm_suggestion_count"]))
            lines.append("- **模型Top-K数量**: {}".format(consistency["model_top_k_count"]))
            lines.append("- **匹配数量**: {}".format(consistency["matched_count"]))
            if consistency["matched_codes"]:
                lines.append("- **匹配代码**: {}".format(", ".join(consistency["matched_codes"])))
            lines.append("- **一致率**: {:.2%}".format(consistency["consistency_rate"]))
            if e3c_is_template:
                lines.append("- **生成方式**: 模板规则（非LLM）")
            lines.append("")
    else:
        lines.append("## E3c：高粒度（DeepSeek LLM完整推荐）")
        lines.append("")
        lines.append("> ⚠️ E3c未生成：DeepSeek API未配置或调用失败。")
        lines.append("")

    return "\n".join(lines)


def generate_experiment_report(
    case1_e3c_success: bool,
    case2_e3c_success: bool,
    case1_consistency: dict[str, Any] | None,
    case2_consistency: dict[str, Any] | None,
    case1_is_template: bool = False,
    case2_is_template: bool = False,
) -> str:
    """
    生成实验3总报告摘要。

    参数:
        case1_e3c_success: 案例1（牛市）是否成功调用E3c
        case2_e3c_success: 案例2（熊市）是否成功调用E3c
        case1_consistency: 案例1一致性评估结果
        case2_consistency: 案例2一致性评估结果
        case1_is_template: 案例1是否为模板生成
        case2_is_template: 案例2是否为模板生成

    返回:
        Markdown格式的实验报告摘要文本
    """
    lines: list[str] = []
    lines.append("# 实验3：AI推荐信息粒度对比 — 总报告")
    lines.append("")
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    lines.append("## 1. 实验设计")
    lines.append("")
    lines.append("| 信息粒度 | 内容 | 目的 |")
    lines.append("|---------|------|-----|")
    lines.append("| E3a | 模型Top-K得分 + 全局Top5因子重要性 | 基线：仅提供排序结果 |")
    lines.append("| E3b | E3a + 每只Top-K股票的Top5特征贡献分解 | 中等：增加个股可解释性 |")
    lines.append("| E3c | E3b + DeepSeek LLM生成完整投资建议 | 高：端到端AI决策辅助 |")
    lines.append("")

    lines.append("## 2. 截面选择")
    lines.append("")
    lines.append("- **牛市截面**: 测试集中周收益率最高的一周")
    lines.append("- **熊市截面**: 测试集中周收益率最低的一周")
    lines.append("")

    lines.append("## 3. 执行结果")
    lines.append("")
    lines.append("| 案例 | E3c状态 | 生成方式 | 一致率 |")
    lines.append("|------|--------|---------|-------|")
    c1_status = "✅ 已生成" if case1_e3c_success else "⚠️ 未生成"
    c2_status = "✅ 已生成" if case2_e3c_success else "⚠️ 未生成"
    c1_method = "模板" if case1_is_template else "LLM"
    c2_method = "模板" if case2_is_template else "LLM"
    c1_cr = f"{case1_consistency['consistency_rate']:.2%}" if case1_consistency else "N/A"
    c2_cr = f"{case2_consistency['consistency_rate']:.2%}" if case2_consistency else "N/A"
    lines.append(f"| 牛市截面 | {c1_status} | {c1_method} | {c1_cr} |")
    lines.append(f"| 熊市截面 | {c2_status} | {c2_method} | {c2_cr} |")
    lines.append("")

    lines.append("## 4. 关键结论")
    lines.append("")
    both_ok = case1_e3c_success and case2_e3c_success
    any_template = case1_is_template or case2_is_template
    if both_ok:
        avg_cr = 0.0
        count = 0
        if case1_consistency:
            avg_cr += case1_consistency["consistency_rate"]
            count += 1
        if case2_consistency:
            avg_cr += case2_consistency["consistency_rate"]
            count += 1
        if count > 0:
            avg_cr /= count
            lines.append(f"- 平均一致率: **{avg_cr:.2%}**")
        if any_template:
            lines.append("- ⚠️ 部分或全部E3c由模板规则生成（DeepSeek API不可用），一致率仅反映模板与模型的重合度，不代表LLM真实表现。")
            lines.append("- API恢复后请重新运行 `python scripts/run_experiment3.py` 以获取真实LLM推荐。")
    else:
        lines.append("- E3c部分或全部未能生成，请检查DeepSeek API配置。")
    lines.append("")

    lines.append("## 5. 信息粒度对比分析")
    lines.append("")
    lines.append("| 维度 | E3a（基础） | E3b（中等） | E3c（高） |")
    lines.append("|------|------------|------------|----------|")
    lines.append("| 信息量 | 模型得分+全局因子 | +个股特征贡献 | +结构化投资建议 |")
    lines.append("| 可解释性 | 低（仅排序） | 中（因子归因） | 高（决策推理） |")
    lines.append("| 可操作性 | 低 | 中 | 高 |")
    lines.append("| 依赖LLM | 否 | 否 | 是 |")
    lines.append("")

    lines.append("## 6. 文件清单")
    lines.append("")
    lines.append("| 文件 | 说明 |")
    lines.append("|------|------|")
    lines.append("| case1_bull.md | 牛市截面完整案例 |")
    lines.append("| case2_bear.md | 熊市截面完整案例 |")
    lines.append("| experiment3_report.md | 本报告 |")
    lines.append("| experiment3_summary.json | 结构化摘要数据 |")
    lines.append("")

    return "\n".join(lines)


def run_single_case(
    predictor: ModelPredictor,
    section_df: pd.DataFrame,
    date_str: str,
    avg_return: float,
    case_label: str,
) -> tuple[str, str, str, dict[str, Any] | None, bool, bool]:
    """
    对单个截面运行完整的E3a/E3b/E3c流程。

    参数:
        predictor: ModelPredictor实例
        section_df: 单个截面的DataFrame
        date_str: 截面日期字符串
        avg_return: 截面平均收益率
        case_label: 案例标签（如 "牛市截面案例"）

    返回:
        (e3a_text, e3b_text, e3c_text, consistency_dict, e3c_success, e3c_is_template)
        e3c_is_template为True时表示E3c由模板生成而非LLM
    """
    logger.info("处理%s: %s, %d只股票", case_label, date_str, len(section_df))

    top_result = predictor.get_top_stocks(section_df, top_n=TOP_K_STOCKS, with_contributions=True)
    top_stocks_list = []
    for _, row in top_result.iterrows():
        top_stocks_list.append({
            "stock_code": str(row["stock_code"]),
            "score": float(row["score"]),
        })

    feature_importance = predictor.get_feature_importance()
    logger.info("Top-K股票已获取，全局特征重要性已计算")

    e3a_text = build_e3a_prompt(date_str, top_stocks_list, feature_importance)
    logger.info("E3a提示词已生成")

    import lightgbm as lgb
    bst = predictor._booster
    assert isinstance(bst, lgb.Booster), "仅支持LightGBM模型的特征贡献分解"

    factor_cols = predictor._factor_cols
    contributions = compute_feature_contribution(bst, section_df, factor_cols, top_k=CONTRIB_TOP_K)
    logger.info("特征贡献分解已完成，%d只股票", len(contributions))

    e3b_text = build_e3b_prompt(e3a_text, top_stocks_list, contributions)
    logger.info("E3b提示词已生成")

    e3c_text, e3c_success = call_deepseek_for_e3c(e3b_text, top_stocks_list)
    e3c_is_template = False

    if not e3c_success:
        logger.info("DeepSeek API不可用，使用模板生成E3c降级方案")
        e3c_text = generate_template_e3c(
            date_str, top_stocks_list, contributions, feature_importance, avg_return,
        )
        e3c_is_template = True
        e3c_success = True
        logger.info("模板化E3c已生成，长度=%d字符", len(e3c_text))

    consistency: dict[str, Any] | None = None
    if e3c_success and e3c_text:
        llm_codes = parse_llm_stock_suggestions(e3c_text)
        model_top_codes = [str(s["stock_code"]) for s in top_stocks_list]
        consistency = calculate_consistency(llm_codes, model_top_codes)
        logger.info(
            "一致性评估完成: LLM建议%d只, 匹配%d只, 一致率=%.2f%%",
            consistency["llm_suggestion_count"],
            consistency["matched_count"],
            consistency["consistency_rate"] * 100,
        )

    return e3a_text, e3b_text, e3c_text, consistency, e3c_success, e3c_is_template


def main() -> None:
    """
    主流程：
    1. 加载测试集数据
    2. 选取牛市/熊市两个典型截面
    3. 加载LightGBM模型并预测
    4. 分别对两个截面执行E3a/E3b/E3c流程
    5. 保存案例文件和总报告
    """
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("步骤1：加载测试集数据")
    logger.info("=" * 60)
    test_df = load_test_data()

    logger.info("=" * 60)
    logger.info("步骤2：选取典型截面（牛市/熊市）")
    logger.info("=" * 60)
    bull_df, bull_date, bull_ret, bear_df, bear_date, bear_ret = find_extreme_sections(test_df)

    logger.info("=" * 60)
    logger.info("步骤3：加载LightGBM模型")
    logger.info("=" * 60)
    predictor = ModelPredictor("lightgbm")
    logger.info("模型加载完成: %s", predictor)

    case1_e3c_success = False
    case2_e3c_success = False
    case1_consistency: dict[str, Any] | None = None
    case2_consistency: dict[str, Any] | None = None
    case1_is_template = False
    case2_is_template = False

    logger.info("=" * 60)
    logger.info("步骤4a：处理牛市截面")
    logger.info("=" * 60)
    try:
        e3a_1, e3b_1, e3c_1, cons_1, ok_1, tpl_1 = run_single_case(
            predictor, bull_df, bull_date, bull_ret, "牛市截面案例",
        )
        case1_e3c_success = ok_1
        case1_consistency = cons_1
        case1_is_template = tpl_1

        report_1 = generate_case_report(
            "牛市截面案例", bull_date, bull_ret,
            e3a_1, e3b_1, e3c_1, cons_1, ok_1, tpl_1,
        )
        case1_path = EXPERIMENT_DIR / "case1_bull.md"
        with open(case1_path, "w", encoding="utf-8") as f:
            f.write(report_1)
        logger.info("牛市截面案例已保存: %s", case1_path)
    except Exception as exc:
        logger.error("牛市截面处理失败: %s", exc)

    logger.info("=" * 60)
    logger.info("步骤4b：处理熊市截面")
    logger.info("=" * 60)
    try:
        e3a_2, e3b_2, e3c_2, cons_2, ok_2, tpl_2 = run_single_case(
            predictor, bear_df, bear_date, bear_ret, "熊市截面案例",
        )
        case2_e3c_success = ok_2
        case2_consistency = cons_2
        case2_is_template = tpl_2

        report_2 = generate_case_report(
            "熊市截面案例", bear_date, bear_ret,
            e3a_2, e3b_2, e3c_2, cons_2, ok_2, tpl_2,
        )
        case2_path = EXPERIMENT_DIR / "case2_bear.md"
        with open(case2_path, "w", encoding="utf-8") as f:
            f.write(report_2)
        logger.info("熊市截面案例已保存: %s", case2_path)
    except Exception as exc:
        logger.error("熊市截面处理失败: %s", exc)

    logger.info("=" * 60)
    logger.info("步骤5：生成实验总报告")
    logger.info("=" * 60)
    final_report = generate_experiment_report(
        case1_e3c_success, case2_e3c_success,
        case1_consistency, case2_consistency,
        case1_is_template, case2_is_template,
    )
    report_path = EXPERIMENT_DIR / "experiment3_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(final_report)
    logger.info("实验总报告已保存: %s", report_path)

    summary: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "bull_section": {
            "date": bull_date,
            "avg_return": bull_ret,
            "e3c_success": case1_e3c_success,
            "e3c_is_template": case1_is_template,
            "consistency": case1_consistency,
        },
        "bear_section": {
            "date": bear_date,
            "avg_return": bear_ret,
            "e3c_success": case2_e3c_success,
            "e3c_is_template": case2_is_template,
            "consistency": case2_consistency,
        },
    }
    summary_path = EXPERIMENT_DIR / "experiment3_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    logger.info("实验摘要已保存: %s", summary_path)

    logger.info("=" * 60)
    logger.info("实验3完成！")
    logger.info("=" * 60)
    logger.info("生成的文件:")
    logger.info("  - %s", EXPERIMENT_DIR / "case1_bull.md")
    logger.info("  - %s", EXPERIMENT_DIR / "case2_bear.md")
    logger.info("  - %s", EXPERIMENT_DIR / "experiment3_report.md")
    logger.info("  - %s", summary_path)
    logger.info("E3c调用状态: 牛市=%s%s, 熊市=%s%s",
                "✅" if case1_e3c_success else "⚠️",
                "(模板)" if case1_is_template else "(LLM)",
                "✅" if case2_e3c_success else "⚠️",
                "(模板)" if case2_is_template else "(LLM)")


if __name__ == "__main__":
    main()
