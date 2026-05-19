"""
最终股票池筛选脚本：综合存续期、ST标记、行业分类、数据完整性、流动性等维度确定最终股票池。

用法：
    cd backend
    python scripts/build_final_stock_pool.py

输出：
    data/meta/final_stock_pool.parquet  - 最终股票池
    data/meta/stock_pool_report.md      - 分析报告
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_fetcher import META_DATA_DIR, _normalize_stock_code, _retry_call

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

START_DATE = "2016-01-01"
END_DATE = "2024-12-31"
DATA_COVERAGE_THRESHOLD = 0.5
LIQUIDITY_PERCENTILE = 5


def step1_load_surviving_stocks() -> pd.DataFrame:
    """第1步：加载存续期筛选后的股票（2016-01-01前上市且未退市）。"""
    from src.stock_pool import get_surviving_stocks

    df = get_surviving_stocks(start_date=START_DATE, end_date=END_DATE)
    logger.info("第1步 - 存续期筛选: %d 只", len(df))
    return df


def step2_filter_delisted(df: pd.DataFrame) -> pd.DataFrame:
    """第2步：排除名称含"退"字的股票，标记含"ST"的股票。"""
    before = len(df)
    df = df[~df["name"].str.contains("退", na=False)].copy()
    df["is_st"] = df["name"].str.contains("ST", case=False, na=False)
    logger.info("第2步 - 排除退市标记: %d -> %d (排除 %d 只)", before, len(df), before - len(df))
    return df


def step3_add_industry(df: pd.DataFrame) -> pd.DataFrame:
    """第3步：获取申万一级行业分类。"""
    # 申万一级行业代码前2位 -> 行业名称映射（申万行业分类2021版）
    SW_FIRST_INDUSTRY_MAP: dict[str, str] = {
        "11": "农林牧渔", "21": "采掘", "22": "采掘", "23": "采掘",
        "24": "化工", "25": "化工", "26": "钢铁", "27": "有色金属",
        "28": "电子", "31": "家用电器", "32": "家用电器",
        "33": "食品饮料", "34": "纺织服饰", "35": "轻工制造",
        "36": "医药生物", "37": "公用事业", "41": "交通运输",
        "42": "房地产", "43": "商贸零售", "44": "商贸零售",
        "45": "社会服务", "46": "银行", "47": "银行",
        "48": "非银金融", "49": "综合", "51": "综合",
        "61": "建筑材料", "62": "建筑装饰", "63": "电力设备",
        "64": "机械设备", "65": "国防军工", "71": "计算机",
        "72": "传媒", "73": "通信", "74": "通信",
        "75": "煤炭", "76": "石油石化", "77": "环保",
    }

    try:
        import akshare as ak

        # 获取个股行业分类历史
        clf_hist = _retry_call(ak.stock_industry_clf_hist_sw)
        if clf_hist is None or clf_hist.empty:
            logger.warning("个股行业分类历史获取失败，使用硬编码映射")
            df["industry"] = "未知"
            return df

        # 取每只股票最新的一级行业分类（industry_code前2位）
        clf_hist = clf_hist.sort_values(["symbol", "start_date"])
        latest = clf_hist.groupby("symbol").last().reset_index()
        latest["industry_first2"] = latest["industry_code"].astype(str).str[:2]
        latest["industry_name"] = latest["industry_first2"].map(SW_FIRST_INDUSTRY_MAP).fillna("未知")

        stock_industry_map = dict(zip(
            latest["symbol"].astype(str),
            latest["industry_name"]
        ))

        df["industry"] = df["code"].map(stock_industry_map).fillna("未知")
        industry_count = (df["industry"] != "未知").sum()
        logger.info("第3步 - 行业分类: %d / %d 只已标注行业", industry_count, len(df))
    except Exception as exc:
        logger.warning("行业分类获取失败: %s，使用'未知'填充", exc)
        df["industry"] = "未知"
    return df


def step4_check_data_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """第4步：检查周频因子数据中的覆盖率。"""
    try:
        data_dir = PROJECT_ROOT / "data"
        parts = []
        for name in ("train", "val", "test"):
            p = data_dir / f"{name}.parquet"
            if p.exists():
                parts.append(pd.read_parquet(p, columns=["stock_code"]))

        if not parts:
            logger.warning("未找到训练/验证/测试数据，跳过覆盖率检查")
            df["data_coverage"] = 1.0
            df["low_coverage"] = False
            return df

        all_data = pd.concat(parts, ignore_index=True)
        total_weeks = all_data.groupby("stock_code").size()
        max_weeks = total_weeks.max() if len(total_weeks) > 0 else 1
        coverage = (total_weeks / max_weeks).to_dict()

        df["data_coverage"] = df["code"].map(coverage).fillna(0.0)
        df["low_coverage"] = df["data_coverage"] < DATA_COVERAGE_THRESHOLD
        low_count = df["low_coverage"].sum()
        logger.info(
            "第4步 - 数据覆盖率: %d 只覆盖率低于 %.0f%%",
            low_count,
            DATA_COVERAGE_THRESHOLD * 100,
        )
    except Exception as exc:
        logger.warning("数据覆盖率检查失败: %s", exc)
        df["data_coverage"] = 1.0
        df["low_coverage"] = False
    return df


def step5_filter_liquidity(df: pd.DataFrame) -> pd.DataFrame:
    """第5步：流动性筛选（基于换手率因子均值，排除低流动性股票）。"""
    try:
        data_dir = PROJECT_ROOT / "data"
        parts = []
        for name in ("train", "val", "test"):
            p = data_dir / f"{name}.parquet"
            if p.exists():
                pf = pd.read_parquet(p)
                cols = ["stock_code"]
                # 优先使用换手率因子
                for col in pf.columns:
                    if "turnover" in col.lower():
                        cols.append(col)
                        break
                if len(cols) > 1:
                    parts.append(pf[cols])

        if not parts or all(len(p.columns) <= 1 for p in parts):
            logger.warning("未找到换手率数据，跳过流动性筛选")
            df["avg_turnover"] = np.nan
            df["low_liquidity"] = False
            return df

        all_data = pd.concat(parts, ignore_index=True)
        turnover_col = [c for c in all_data.columns if c != "stock_code"][0]
        avg_turnover = all_data.groupby("stock_code")[turnover_col].mean()

        df["avg_turnover"] = df["code"].map(avg_turnover)
        valid_turnover = df["avg_turnover"].dropna()
        if valid_turnover.empty or valid_turnover.std() < 1e-10:
            logger.warning("换手率数据无效（全为0或无变化），跳过流动性筛选")
            df["low_liquidity"] = False
            return df

        threshold = valid_turnover.quantile(LIQUIDITY_PERCENTILE / 100.0)
        df["low_liquidity"] = df["avg_turnover"].notna() & (df["avg_turnover"] < threshold)

        before = len(df)
        df = df[~df["low_liquidity"]].copy()
        logger.info(
            "第5步 - 流动性筛选: %d -> %d (排除 %d 只, 换手率阈值=%.4f)",
            before,
            len(df),
            before - len(df),
            threshold,
        )
    except Exception as exc:
        logger.warning("流动性筛选失败: %s，跳过", exc)
        df["avg_turnover"] = np.nan
        df["low_liquidity"] = False
    return df


def generate_report(
    df: pd.DataFrame,
    old_pool_size: int,
    funnel: dict[str, int],
) -> str:
    """生成Markdown格式的股票池分析报告。"""
    lines: list[str] = []
    lines.append("# 股票池筛选分析报告")
    lines.append("")
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    lines.append("## 1. 筛选条件")
    lines.append("")
    lines.append("| 步骤 | 条件 | 说明 |")
    lines.append("|------|------|------|")
    lines.append(f"| 存续期 | 上市日期 < {START_DATE} | 近10年存续 |")
    lines.append('| 退市排除 | 名称不含"退"字 | 排除已退市标的 |')
    lines.append('| ST标记 | 名称含"ST" | 仅标记不排除 |')
    lines.append("| 行业分类 | 申万一级行业 | 来源: akshare |")
    lines.append(f"| 数据覆盖 | 覆盖率 >= {DATA_COVERAGE_THRESHOLD * 100:.0f}% | 标记低覆盖率 |")
    lines.append(f"| 流动性 | 成交额 > P{LIQUIDITY_PERCENTILE} | 排除低流动性 |")
    lines.append("")

    lines.append("## 2. 筛选漏斗（各步骤股票数量变化）")
    lines.append("")
    lines.append("| 步骤 | 股票数量 | 变化 |")
    lines.append("|------|---------|------|")
    prev = 0
    for step_name, count in funnel.items():
        if prev > 0:
            delta = f"{count - prev:+d}" if count != prev else "0"
        else:
            delta = ""
        lines.append(f"| {step_name} | {count} | {delta} |")
        prev = count
    lines.append("")

    lines.append("## 3. 最终股票池规模")
    lines.append("")
    lines.append(f"**最终股票池: {len(df)} 只**")
    lines.append("")
    st_count = df["is_st"].sum() if "is_st" in df.columns else 0
    lines.append(f"- ST股票: {st_count} 只")
    low_cov = df["low_coverage"].sum() if "low_coverage" in df.columns else 0
    lines.append(f"- 低数据覆盖率: {low_cov} 只")
    lines.append("")

    if "industry" in df.columns:
        lines.append("## 4. 申万一级行业分布")
        lines.append("")
        industry_counts = df["industry"].value_counts()
        lines.append("| 行业 | 股票数量 | 占比 |")
        lines.append("|------|---------|------|")
        for ind, cnt in industry_counts.items():
            pct = cnt / len(df) * 100
            lines.append(f"| {ind} | {cnt} | {pct:.1f}% |")
        lines.append("")

    if "listing_date" in df.columns:
        lines.append("## 5. 上市日期分布")
        lines.append("")
        df_copy = df.copy()
        df_copy["listing_year"] = pd.to_datetime(df_copy["listing_date"]).dt.year
        year_counts = df_copy["listing_year"].value_counts().sort_index()
        lines.append("| 年份 | 股票数量 |")
        lines.append("|------|---------|")
        for year, cnt in year_counts.items():
            lines.append(f"| {year} | {cnt} |")
        lines.append("")

    lines.append("## 6. 与旧股票池对比")
    lines.append("")
    lines.append("| 指标 | 旧股票池 (2014起) | 新股票池 (2016起) |")
    lines.append("|------|-----------------|-----------------|")
    lines.append(f"| 起始日期 | 2014-01-01 | {START_DATE} |")
    lines.append(f"| 股票数量 | {old_pool_size} | {len(df)} |")
    lines.append(f"| 差异 | - | +{len(df) - old_pool_size} |")
    lines.append("")

    if "data_coverage" in df.columns:
        lines.append("## 7. 数据覆盖率统计")
        lines.append("")
        cov = df["data_coverage"]
        lines.append("| 统计量 | 值 |")
        lines.append("|--------|-----|")
        lines.append(f"| 均值 | {cov.mean():.4f} |")
        lines.append(f"| 中位数 | {cov.median():.4f} |")
        lines.append(f"| 最小值 | {cov.min():.4f} |")
        lines.append(f"| 最大值 | {cov.max():.4f} |")
        lines.append(f"| 覆盖率>=90% | {(cov >= 0.9).sum()} 只 |")
        lines.append(f"| 覆盖率>=50% | {(cov >= 0.5).sum()} 只 |")
        lines.append(f"| 覆盖率<50% | {(cov < 0.5).sum()} 只 |")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    """主流程：逐步筛选并生成报告。"""
    funnel: dict[str, int] = {}

    old_pool_path = META_DATA_DIR / "surviving_stocks_20140101_20241231.parquet"
    old_pool_size = len(pd.read_parquet(old_pool_path)) if old_pool_path.exists() else 2258

    df = step1_load_surviving_stocks()
    funnel["存续期筛选 (2016前上市)"] = len(df)

    df = step2_filter_delisted(df)
    funnel["排除退市标记"] = len(df)

    df = step3_add_industry(df)
    funnel["行业分类"] = len(df)

    df = step4_check_data_coverage(df)
    funnel["数据覆盖率检查"] = len(df)

    df = step5_filter_liquidity(df)
    funnel["流动性筛选"] = len(df)

    out_path = META_DATA_DIR / "final_stock_pool.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("最终股票池已保存: %s (%d 只)", out_path, len(df))

    report = generate_report(df, old_pool_size, funnel)
    report_path = META_DATA_DIR / "stock_pool_report.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info("分析报告已保存: %s", report_path)

    print(f"\n最终股票池: {len(df)} 只")
    print(f"文件: {out_path}")
    print(f"报告: {report_path}")


if __name__ == "__main__":
    main()
