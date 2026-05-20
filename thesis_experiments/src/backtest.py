# [共享文件] 本文件同时存在于 project/backend/src/ 和 thesis_experiments/src/，修改时请同步更新两处
"""
周频滚动回测引擎：固定加载已训练模型，按周五截面调仓，显式交易成本与涨跌停约束。

用法::

    python -m src.backtest              # 快速打印测试集回测首尾
    python -m src.backtest --run        # 跑双模型、写 SQLite、导出 data/backtest_results/*.json
    python -m src.backtest --compare    # 双模型对比：comparison.json、comparison_nav.json、可选 Plotly 报告
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm as _tqdm_bar
except ImportError:

    def _tqdm_bar(iterable, **_kw):  # type: ignore[no-untyped-def]
        return iterable

try:
    import akshare as ak
except ImportError:
    ak = None  # type: ignore[assignment]


from .config import get_settings
from .data_loader import DATA_OUT_DIR, add_future_return, split_by_time
from .db import init_db, session_scope
from .db.models import BacktestResult
from .metrics import aggregate_metrics, metrics_to_json, weekly_nav_to_daily_business_ffill
from .predictor import ModelKind, ModelPredictor

logger = logging.getLogger(__name__)


def _score_panel_with_fallback(
    pred: Any,
    panel_df: pd.DataFrame,
) -> tuple[dict[pd.Timestamp, pd.DataFrame], set[pd.Timestamp]]:
    """
    优先整表 predict_panel；失败则按调仓周逐次 predict。
    返回 {date -> 该日 score 表}，以及预测失败需跳过调仓的日期集合。
    """
    failed: set[pd.Timestamp] = set()
    if panel_df.empty:
        return {}, failed

    scored_full: pd.DataFrame | None = None
    try:
        if hasattr(pred, "predict_panel"):
            scored_full = pred.predict_panel(panel_df)
    except Exception as exc:
        logger.warning("批量预测失败，按调仓周拆分重试: %s", exc)
        scored_full = None

    if scored_full is None:
        parts: list[pd.DataFrame] = []
        for dt, g in panel_df.groupby("date", sort=True):
            try:
                fac = g[["stock_code"] + [c for c in pred._factor_cols if c in g.columns]]
                if fac.empty:
                    failed.add(pd.Timestamp(dt))
                    continue
                sc = pred.predict(fac)
                parts.append(
                    pd.DataFrame(
                        {
                            "date": pd.Timestamp(dt),
                            "stock_code": sc["stock_code"].astype(str).to_numpy(),
                            "score": sc["score"].to_numpy(dtype=np.float64),
                        }
                    )
                )
            except Exception as exc2:
                logger.warning("调仓日 %s 预测失败，跳过本周调仓（保持持仓）: %s", dt, exc2)
                failed.add(pd.Timestamp(dt))
        scored_full = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["date", "stock_code", "score"])

    by_date: dict[pd.Timestamp, pd.DataFrame] = {}
    if not scored_full.empty:
        for d, g in scored_full.groupby("date", sort=False):
            ts = pd.Timestamp(d)
            sub = g[["stock_code", "score"]].copy()
            if sub["score"].isna().all():
                failed.add(ts)
                continue
            by_date[ts] = sub
    return by_date, failed


def _select_target_codes(
    day_scores: pd.DataFrame,
    px: pd.DataFrame,
    top_n: int,
    enable_limit_price: bool = False,
    reverse_sort: bool = False,
) -> list[str]:
    """
    向量化合并涨跌停掩码后取 score Top N 或 Bottom N（可买）。

    参数:
        day_scores: 当期模型打分结果，需包含 stock_code 和 score 列
        px: 当期截面数据，索引为股票代码，需包含 buy_blocked_limit_up 列
        top_n: 选股数量上限
        enable_limit_price: 是否启用涨跌停约束；为 False 时跳过涨停过滤，直接按 score 排序取 Top N
        reverse_sort: 是否按分数升序排列选股（选 Bottom N）；默认 False（降序选 Top N）

    返回:
        目标持仓股票代码列表
    """
    if day_scores.empty:
        return []
    if not enable_limit_price:
        merged = day_scores.sort_values("score", ascending=reverse_sort, kind="mergesort")
        return merged["stock_code"].astype(str).head(top_n).tolist()
    pxr = px.reset_index()
    merged = day_scores.merge(
        pxr[["stock_code", "buy_blocked_limit_up"]],
        on="stock_code",
        how="inner",
    )
    merged = merged.loc[~merged["buy_blocked_limit_up"]]
    merged = merged.sort_values("score", ascending=reverse_sort, kind="mergesort")
    return merged["stock_code"].astype(str).head(top_n).tolist()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
BACKTEST_RESULTS_DIR = DATA_DIR / "backtest_results"
REPORTS_DIR = PROJECT_ROOT / "reports"

# 普通主板涨跌幅简化：10%（科创板/创业板 20% 未区分，与任务说明一致）
LIMIT_UP_MULT = 1.10
LIMIT_DOWN_MULT = 0.90


class BacktestEngine:
    """
    周频回测：每周五（数据中的 date 截面）用当期因子与固定模型打分，持有至下一调仓日。

    交易规则（均在代码中显式实现，便于审计与复现）：
    - T+1：A 股当日买入不可当日卖出；本引擎调仓周期为周，卖出发生在下一调仓周，
      与「买入周」不同，故不存在日内回转，任务要求下可忽略日内 T+1。
    - 佣金 commission：买卖双边按成交额的比率从现金扣除（默认万三 0.0003）。
    - 滑点 slippage：买入成交价=收盘价*(1+slippage)，卖出=收盘价*(1-slippage)（双边默认 0.1%）。
    - 印花税 stamp_tax：仅卖出侧，按卖出成交额比率（默认万五 0.0005），在佣金之外扣除。
    - 涨跌停：涨停价=前一周收盘价*LIMIT_UP_MULT，跌停价=前一周收盘价*LIMIT_DOWN_MULT；
      涨停时视为无法买入；跌停时视为无法卖出。价格序列无日线时，用周收盘价与前一周收盘价比较。
    - 空仓：目标 Top N 全部因涨停无法买入，或剩余现金不足以按规则成交任何一笔加仓时，
      保留现金；若已有持仓则仍按可卖尽卖、可买尽买处理。
    """

    def __init__(
        self,
        model_type: ModelKind,
        top_n: int,
        initial_capital: float = 1_000_000.0,
        commission: float = 0.0003,
        slippage: float = 0.001,
        stamp_tax: float = 0.0005,
        enable_limit_price: bool = False,
        reverse_sort: bool = False,
        vol_penalty: float = 0.0,
        rebalance_freq: int = 1,
        custom_predictor: ModelPredictor | None = None,
        *,
        data_dir: Path | None = None,
    ) -> None:
        """
        初始化回测引擎。

        参数:
            model_type: 模型类型，"lightgbm" 或 "xgboost"
            top_n: 每期选股数量，至少为 1
            initial_capital: 初始资金，必须为正
            commission: 买卖双边佣金率，默认万三
            slippage: 双边滑点比率，默认 0.1%
            stamp_tax: 卖出印花税率，默认万五
            enable_limit_price: 是否启用涨跌停约束，默认 False（不启用）
            reverse_sort: 是否按模型分数升序排列选股（选 Bottom N），默认 False（降序选 Top N）
            vol_penalty: 波动率惩罚系数，默认 0.0（不调整）；大于 0 时，选股分数将减去
                vol_penalty * volatility_12w，降低高波动股票的选中概率
            rebalance_freq: 调仓频率（单位：周），默认 1（每周调仓）；大于 1 时仅在调仓周
                （第0周、第rebalance_freq周、第2*rebalance_freq周...）执行选股与调仓，
                非调仓周保持上一期持仓不变，仅更新持仓市值和现金
            custom_predictor: 自定义预测器（如 FusionPredictor），默认 None 时使用 ModelPredictor；
                传入后 predict_panel / predict 接口需与 ModelPredictor 一致
            data_dir: 数据目录路径，默认使用配置或 DATA_OUT_DIR

        异常:
            ValueError: top_n 小于 1、initial_capital 非正或 rebalance_freq 小于 1 时抛出
        """
        if top_n < 1:
            raise ValueError("top_n 至少为 1")
        if initial_capital <= 0:
            raise ValueError("initial_capital 必须为正")
        if rebalance_freq < 1:
            raise ValueError("rebalance_freq 至少为 1")
        self.model_type: ModelKind = model_type
        self.top_n = top_n
        self.initial_capital = float(initial_capital)
        self.commission = float(commission)
        self.slippage = float(slippage)
        self.stamp_tax = float(stamp_tax)
        self.enable_limit_price = enable_limit_price
        self.reverse_sort = reverse_sort
        self.vol_penalty = float(vol_penalty)
        self._rebalance_freq = int(rebalance_freq)
        self.custom_predictor = custom_predictor
        self._last_known_price: dict[str, float] = {}
        s = get_settings()
        self._data_dir = (
            Path(data_dir)
            if data_dir is not None
            else (Path(s.quant_data_dir) if s.quant_data_dir.strip() else DATA_OUT_DIR)
        )

    def _buy_cash_per_share(self, close: float) -> float:
        """买入 1 股所需现金：含滑点后的单价 * (1+佣金)。"""
        unit = float(close) * (1.0 + self.slippage)
        return unit * (1.0 + self.commission)

    def _cash_from_sell(self, shares: float, close: float) -> float:
        """卖出 shares 后入账现金：含滑点、佣金、印花税。"""
        if shares <= 0:
            return 0.0
        unit = float(close) * (1.0 - self.slippage)
        gross = float(shares) * unit
        return gross * (1.0 - self.commission - self.stamp_tax)

    def load_weekly_data(
        self,
        path: Path | None = None,
        *,
        concat_splits: bool = True,
    ) -> pd.DataFrame:
        """
        加载周频截面：日期、股票代码、因子、未来一周收益率等；按时间升序排列。

        - 若提供 path：读取该 parquet（需含 date、stock_code；若有 close 则直接使用，否则需有
          future_return_1w 以便递推合成收盘价）。
        - 否则 concat_splits=True 时合并 data_dir 下 train/val/test.parquet（与任务 2 落盘一致）。
        """
        if path is not None:
            pp = Path(path)
            with pp.open("rb") as fp:
                raw = pd.read_parquet(fp)
        elif concat_splits:
            parts: list[pd.DataFrame] = []
            for name in ("train", "val", "test"):
                p = self._data_dir / f"{name}.parquet"
                if not p.exists():
                    raise FileNotFoundError(str(p))
                with p.open("rb") as fp:
                    parts.append(pd.read_parquet(fp))
            raw = pd.concat(parts, ignore_index=True)
        else:
            raise ValueError("path 为 None 时需要 concat_splits=True")

        date_col = "date" if "date" in raw.columns else "日期"
        code_col = "stock_code" if "stock_code" in raw.columns else "股票代码"
        if date_col not in raw.columns or code_col not in raw.columns:
            raise KeyError("周频数据需包含 date 与 stock_code（或中文列名）")

        out = raw.copy()
        out["date"] = pd.to_datetime(out[date_col])
        out["stock_code"] = out[code_col].astype(str)
        out = out.sort_values(["date", "stock_code"]).reset_index(drop=True)

        if "future_return_1w" not in out.columns:
            close_src = "close" if "close" in out.columns else ("收盘" if "收盘" in out.columns else None)
            if close_src is None:
                raise KeyError("缺少 future_return_1w，且无法从收盘价构造标签")
            tmp = out.rename(columns={close_src: "close"}).copy()
            out = add_future_return(tmp, forward_weeks=1)

        if "close" not in out.columns and "收盘" not in out.columns:
            out["close"] = _synthetic_close_from_weekly_returns(out)
        else:
            ccol = "close" if "close" in out.columns else "收盘"
            out["close"] = pd.to_numeric(out[ccol], errors="coerce")

        # 向量化：前收、涨跌停参考价（按股票时间 shift）
        out = out.sort_values(["stock_code", "date"])
        out["prev_close"] = out.groupby("stock_code", sort=False)["close"].shift(1)
        out["limit_up_price"] = out["prev_close"] * LIMIT_UP_MULT
        out["limit_down_price"] = out["prev_close"] * LIMIT_DOWN_MULT
        tol = 1e-9
        pc = out["prev_close"]
        out["buy_blocked_limit_up"] = pc.notna() & (out["close"] >= out["limit_up_price"] - tol)
        out["sell_blocked_limit_down"] = pc.notna() & (out["close"] <= out["limit_down_price"] + tol)

        if not self.enable_limit_price:
            out["buy_blocked_limit_up"] = False
            out["sell_blocked_limit_down"] = False

        out = out.sort_values(["date", "stock_code"]).reset_index(drop=True)
        return out

    def run_backtest(
        self,
        weekly_df: pd.DataFrame | None = None,
        *,
        predictor: ModelPredictor | None = None,
        use_split: Literal["test", "all"] = "test",
        train_end: str = "2020-12-31",
        val_end: str = "2022-12-31",
        skip_initial_weeks: int = 0,
    ) -> pd.DataFrame:
        """
        主回测流程：每周五截面打分、卖出非目标、再买入目标，记录调仓日净值。

        时点说明（避免未来信息）：date=t 行的因子与收盘价视为周五 t 收盘可知；模型得分用于
        决定从 t 收盘起持有至下一调仓周。标签 future_return_1w 仅用于数据管道，回测下单不使用。

        多周调仓说明：当引擎 rebalance_freq > 1 时，仅在调仓周（第0周、第rebalance_freq周、
        第2*rebalance_freq周...）执行选股与调仓；非调仓周保持上一期持仓不变，仅使用当周收盘价
        更新持仓市值和现金快照，不产生任何交易。

        返回列：date, total_value, cash, holdings（JSON 字符串，股票代码 -> 持仓股数）。

        skip_initial_weeks：在选定样本内跳过前若干调仓周，用于因子滚动窗口尚未就绪、或
        与训练集尾部隔离等场景；默认 0 表示从该样本第一个截面开始回测。

        说明：数据为周频时，每行对应一次调仓日（通常为周五）收盘后的持仓与现金快照；
        非自然日逐日净值；若需日度曲线需接入日频行情并扩展估值循环。流程无随机步骤，可复现。

        性能：整段样本先 ``predict_panel`` 批量打分（模型固定、无未来信息），再逐周撮合；
        批量失败时自动按周 ``predict`` 回退。某周预测失败或截面缺列时跳过调仓、保持持仓。
        进度条依赖 tqdm（未安装则静默遍历）。
        """
        if weekly_df is None:
            weekly_df = self.load_weekly_data()

        df = weekly_df.copy()
        if use_split == "test":
            _, _, test_part = split_by_time(df, train_end=train_end, val_end=val_end)
            if test_part.empty:
                raise ValueError("测试集为空，请检查 train_end/val_end 或数据区间")
            df = test_part.sort_values(["date", "stock_code"]).reset_index(drop=True)

        pred = predictor or self.custom_predictor or ModelPredictor(self.model_type, data_dir=self._data_dir)
        factor_cols = pred._factor_cols

        dates = sorted(df["date"].unique())
        if skip_initial_weeks < 0:
            raise ValueError("skip_initial_weeks 不能为负")
        if skip_initial_weeks:
            dates = dates[int(skip_initial_weeks) :]
        if not dates:
            raise ValueError("skip_initial_weeks 过大或样本无调仓日")
        dates_ts = [pd.Timestamp(d) for d in dates]

        # 按日期索引截面，避免每周全表过滤
        date_to_day: dict[pd.Timestamp, pd.DataFrame] = {
            pd.Timestamp(d): g for d, g in df.groupby("date", sort=False)
        }

        # 批预测：整段样本一次推理（模型固定），失败则按周回退；无未来信息（每行仅用当期因子）
        factor_in_df = [c for c in factor_cols if c in df.columns]
        panel_cols = ["date", "stock_code", *factor_in_df]
        panel_parts: list[pd.DataFrame] = []
        for dt in dates_ts:
            g = date_to_day.get(dt)
            if g is None or g.empty:
                continue
            panel_parts.append(g[panel_cols].copy())
        panel_df = pd.concat(panel_parts, ignore_index=True) if panel_parts else pd.DataFrame(columns=["date", "stock_code"])
        scored_by_date, failed_dates = _score_panel_with_fallback(pred, panel_df)
        del panel_df

        cash = self.initial_capital
        positions: dict[str, float] = {}
        rows: list[dict[str, object]] = []

        for week_idx, dt in enumerate(_tqdm_bar(dates_ts, desc="回测", unit="周")):
            day = date_to_day.get(dt)
            if day is None or day.empty:
                continue

            px = day.set_index("stock_code")
            required = ["close", "buy_blocked_limit_up", "sell_blocked_limit_down"]
            missing = [c for c in required if c not in px.columns]
            if missing:
                logger.warning("截面 %s 缺少列 %s，跳过本周调仓", dt, missing)

            for code in px.index:
                self._last_known_price[code] = float(px.loc[code, "close"])

            for code in list(positions.keys()):
                if code not in px.index:
                    continue

            if self._rebalance_freq > 1 and week_idx % self._rebalance_freq != 0:
                m2m = _portfolio_market_value(positions, px["close"], self._last_known_price)
                total = cash + m2m
                rows.append(
                    {
                        "date": dt,
                        "total_value": float(total),
                        "cash": float(cash),
                        "holdings": _holdings_json(positions),
                    }
                )
                continue

            skip_trade = bool(missing) or dt in failed_dates or dt not in scored_by_date
            if skip_trade:
                m2m = _portfolio_market_value(positions, px["close"], self._last_known_price)
                total = cash + m2m
                rows.append(
                    {
                        "date": dt,
                        "total_value": float(total),
                        "cash": float(cash),
                        "holdings": _holdings_json(positions),
                    }
                )
                continue

            day_scores = scored_by_date[dt]

            if self.vol_penalty > 0 and "volatility_12w" in px.columns:
                vol_map = px["volatility_12w"].to_dict()
                day_scores = day_scores.copy()
                day_scores["score"] = day_scores["score"] - self.vol_penalty * day_scores["stock_code"].map(vol_map).fillna(0.0)

            target = _select_target_codes(day_scores, px, self.top_n, self.enable_limit_price, self.reverse_sort)

            for code in list(positions.keys()):
                if code in target:
                    continue
                if code not in px.index:
                    continue
                if bool(px.loc[code, "sell_blocked_limit_down"]):
                    continue
                sh = float(positions.pop(code))
                cl = float(px.loc[code, "close"])
                cash += self._cash_from_sell(sh, cl)

            m2m = _portfolio_market_value(positions, px["close"], self._last_known_price)
            equity = cash + m2m

            if not target:
                rows.append(
                    {
                        "date": dt,
                        "total_value": float(equity),
                        "cash": cash,
                        "holdings": _holdings_json(positions),
                    }
                )
                continue

            stuck_val = _stuck_non_target_value(positions, target, px["close"])
            deployable = max(0.0, equity - stuck_val)
            per_target = deployable / float(len(target))

            for code in target:
                if code not in px.index:
                    continue
                cl = float(px.loc[code, "close"])
                cur_sh = float(positions.get(code, 0.0))
                cur_val = cur_sh * cl
                if cur_val <= per_target + 1e-6:
                    continue
                if bool(px.loc[code, "sell_blocked_limit_down"]):
                    continue
                tgt_sh = per_target / cl
                delta = cur_sh - tgt_sh
                if delta <= 1e-12:
                    continue
                positions[code] = cur_sh - delta
                if positions[code] <= 1e-12:
                    del positions[code]
                cash += self._cash_from_sell(delta, cl)

            m2m = _portfolio_market_value(positions, px["close"], self._last_known_price)
            equity = cash + m2m
            stuck_val = _stuck_non_target_value(positions, target, px["close"])
            deployable = max(0.0, equity - stuck_val)
            per_target = deployable / float(len(target))

            for code in target:
                if code not in px.index:
                    continue
                cl = float(px.loc[code, "close"])
                if bool(px.loc[code, "buy_blocked_limit_up"]):
                    continue
                cur_sh = float(positions.get(code, 0.0))
                cur_val = cur_sh * cl
                need = per_target - cur_val
                if need <= 1e-6:
                    continue
                cps = self._buy_cash_per_share(cl)
                max_cash_buy = cash / cps if cps > 0 else 0.0
                buy_sh = min(need / cl, max_cash_buy)
                if buy_sh <= 1e-12:
                    continue
                cash -= buy_sh * cps
                positions[code] = cur_sh + buy_sh

            m2m = _portfolio_market_value(positions, px["close"], self._last_known_price)
            total = cash + m2m
            rows.append(
                {
                    "date": dt,
                    "total_value": float(total),
                    "cash": float(cash),
                    "holdings": _holdings_json(positions),
                }
            )

        del scored_by_date
        result = pd.DataFrame(rows)
        if result.empty:
            return result

        # 添加基准净值曲线
        if not result.empty:
            dates_min = result["date"].min()
            dates_max = result["date"].max()
            try:
                hs300_nav = _load_benchmark_nav(dates_min, dates_max, "000300")
                zz500_nav = _load_benchmark_nav(dates_min, dates_max, "000905")
                result["benchmark_hs300"] = result["date"].map(
                    lambda d: hs300_nav.get(pd.Timestamp(d), float("nan"))
                )
                result["benchmark_zz500"] = result["date"].map(
                    lambda d: zz500_nav.get(pd.Timestamp(d), float("nan"))
                )
                # 前向填充缺失值（基准周频日期可能不完全对齐）
                result["benchmark_hs300"] = result["benchmark_hs300"].ffill()
                result["benchmark_zz500"] = result["benchmark_zz500"].ffill()
            except Exception as exc:
                logger.warning("基准净值获取失败: %s", exc)
                result["benchmark_hs300"] = float("nan")
                result["benchmark_zz500"] = float("nan")

        # 计算周度收益率
        result["weekly_return"] = result["total_value"].pct_change(fill_method=None)
        result.loc[result.index[0], "weekly_return"] = 0.0

        # 计算相对沪深300的超额收益
        hs300_weekly = result["benchmark_hs300"].pct_change(fill_method=None)
        result["excess_return_hs300"] = result["weekly_return"] - hs300_weekly
        result.loc[result.index[0], "excess_return_hs300"] = 0.0

        # 计算相对中证500的超额收益
        zz500_weekly = result["benchmark_zz500"].pct_change(fill_method=None)
        result["excess_return_zz500"] = result["weekly_return"] - zz500_weekly
        result.loc[result.index[0], "excess_return_zz500"] = 0.0

        return result.sort_values("date").reset_index(drop=True)


def _load_benchmark_nav(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    symbol: str = "000300",
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> pd.Series:
    """
    获取指数周频累计净值（起始=1.0），依次尝试：akshare → baostock → 本地数据回退。

    内置重试机制和本地缓存，提高网络不稳定环境下的获取成功率。

    参数:
        start_date: 起始日期
        end_date: 结束日期
        symbol: 指数代码，"000300"为沪深300，"000905"为中证500
        max_retries: 最大重试次数，默认3
        retry_delay: 重试间隔秒数，默认5.0

    返回:
        以日期为索引的累计净值Series；所有来源均失败时返回空Series
    """
    cache_path = DATA_DIR / "benchmark_cache" / f"{symbol}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.parquet"
    if cache_path.exists():
        try:
            cached = pd.read_parquet(cache_path)
            dates = pd.to_datetime(cached["date"])
            nav = cached["nav"].astype(float)
            logger.info("从本地缓存加载基准净值（symbol=%s）", symbol)
            return pd.Series(nav.to_numpy(), index=pd.DatetimeIndex(dates), dtype=float)
        except Exception:
            pass

    result = _load_benchmark_via_akshare(start_date, end_date, symbol, max_retries, retry_delay)
    if not result.empty:
        _save_benchmark_cache(cache_path, result)
        return result

    result = _load_benchmark_via_baostock(start_date, end_date, symbol, max_retries, retry_delay)
    if not result.empty:
        _save_benchmark_cache(cache_path, result)
        return result

    result = _load_benchmark_from_local(start_date, end_date, symbol)
    if not result.empty:
        _save_benchmark_cache(cache_path, result)
        return result

    logger.warning("基准净值获取失败（symbol=%s），所有来源均不可用", symbol)
    return pd.Series(dtype=float)


def _save_benchmark_cache(cache_path: Path, nav_series: pd.Series) -> None:
    """将基准净值Series保存到本地缓存。"""
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_df = pd.DataFrame({"date": nav_series.index, "nav": nav_series.to_numpy()})
        cache_df.to_parquet(cache_path, index=False)
        logger.info("基准净值已缓存（path=%s）", cache_path.name)
    except Exception:
        pass


def _load_benchmark_via_akshare(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    symbol: str,
    max_retries: int,
    retry_delay: float,
) -> pd.Series:
    """通过akshare获取指数周频净值。"""
    if ak is None:
        logger.warning("akshare 未安装，跳过akshare获取（symbol=%s）", symbol)
        return pd.Series(dtype=float)
    import time as _time
    for attempt in range(1, max_retries + 1):
        try:
            df = ak.index_zh_a_hist(
                symbol=symbol,
                period="weekly",
                start_date=start_date.strftime("%Y%m%d"),
                end_date=end_date.strftime("%Y%m%d"),
            )
            if df is None or df.empty:
                logger.warning("akshare 返回空数据（symbol=%s，第%d次尝试）", symbol, attempt)
                if attempt < max_retries:
                    _time.sleep(retry_delay)
                continue
            dcol = "日期" if "日期" in df.columns else "date"
            ccol = "收盘" if "收盘" in df.columns else ("close" if "close" in df.columns else None)
            if ccol is None:
                logger.warning("基准数据缺少收盘列（symbol=%s）", symbol)
                return pd.Series(dtype=float)
            close = pd.to_numeric(df[ccol], errors="coerce")
            dates = pd.to_datetime(df[dcol])
            nav = close / close.iloc[0]
            logger.info("akshare获取基准净值成功（symbol=%s，%d条）", symbol, len(nav))
            return pd.Series(nav.to_numpy(), index=pd.DatetimeIndex(dates), dtype=float)
        except Exception as exc:
            logger.warning("akshare获取失败（symbol=%s，第%d次尝试）: %s", symbol, attempt, exc)
            if attempt < max_retries:
                _time.sleep(retry_delay)
    logger.warning("akshare获取基准净值失败（symbol=%s），已重试%d次", symbol, max_retries)
    return pd.Series(dtype=float)


def _load_benchmark_via_baostock(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    symbol: str,
    max_retries: int,
    retry_delay: float,
) -> pd.Series:
    """通过baostock获取指数周频净值作为akshare的回退方案。"""
    try:
        import baostock as bs
    except ImportError:
        logger.warning("baostock 未安装，跳过baostock获取（symbol=%s）", symbol)
        return pd.Series(dtype=float)
    import time as _time
    bs_symbol = f"sh.{symbol}" if not symbol.startswith("sh.") else symbol
    for attempt in range(1, max_retries + 1):
        try:
            lg = bs.login()
            if lg.error_code != "0":
                logger.warning("baostock登录失败（第%d次尝试）: %s", attempt, lg.error_msg)
                if attempt < max_retries:
                    _time.sleep(retry_delay)
                continue
            rs = bs.query_history_k_data_plus(
                bs_symbol,
                "date,close",
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                frequency="w",
            )
            rows: list[list[str]] = []
            while rs.error_code == "0" and rs.next():
                rows.append(rs.get_row_data())
            bs.logout()
            if not rows:
                logger.warning("baostock返回空数据（symbol=%s，第%d次尝试）", symbol, attempt)
                if attempt < max_retries:
                    _time.sleep(retry_delay)
                continue
            df = pd.DataFrame(rows, columns=["date", "close"])
            close = pd.to_numeric(df["close"], errors="coerce")
            dates = pd.to_datetime(df["date"])
            valid = close.notna() & dates.notna()
            close = close[valid].reset_index(drop=True)
            dates = dates[valid].reset_index(drop=True)
            if close.empty:
                logger.warning("baostock数据无有效值（symbol=%s）", symbol)
                return pd.Series(dtype=float)
            nav = close / close.iloc[0]
            logger.info("baostock获取基准净值成功（symbol=%s，%d条）", symbol, len(nav))
            return pd.Series(nav.to_numpy(), index=pd.DatetimeIndex(dates), dtype=float)
        except Exception as exc:
            logger.warning("baostock获取失败（symbol=%s，第%d次尝试）: %s", symbol, attempt, exc)
            try:
                bs.logout()
            except Exception:
                pass
            if attempt < max_retries:
                _time.sleep(retry_delay)
    logger.warning("baostock获取基准净值失败（symbol=%s），已重试%d次", symbol, max_retries)
    return pd.Series(dtype=float)


def _load_benchmark_from_local(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    symbol: str,
) -> pd.Series:
    """
    从本地因子数据计算等权市场基准净值作为最终回退方案。

    使用test.parquet中的future_return_1w列，按日期计算全市场等权平均收益率，
    再累乘为净值曲线。注意：此基准为全A股等权基准，并非沪深300/中证500指数，
    仅在网络不可用时作为参考基准。

    参数:
        start_date: 起始日期
        end_date: 结束日期
        symbol: 指数代码（仅用于日志标识）

    返回:
        以日期为索引的累计净值Series；本地数据不可用时返回空Series
    """
    for fname in ("test.parquet", "test.parquet.gz"):
        fpath = DATA_DIR / fname
        if fpath.exists():
            try:
                df = pd.read_parquet(fpath, columns=["date", "future_return_1w"])
                df["date"] = pd.to_datetime(df["date"])
                df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
                if df.empty:
                    continue
                mkt_ret = df.groupby("date")["future_return_1w"].mean().sort_index()
                nav = (1 + mkt_ret).cumprod()
                nav.iloc[:] = nav.to_numpy()
                logger.info(
                    "本地数据回退：使用等权市场基准（symbol=%s，%d条，来源=%s）",
                    symbol, len(nav), fname,
                )
                return nav
            except Exception as exc:
                logger.warning("本地数据回退失败（symbol=%s，%s）: %s", symbol, fname, exc)
                continue
    logger.warning("本地数据回退不可用（symbol=%s），无可用test.parquet", symbol)
    return pd.Series(dtype=float)


def _holdings_json(positions: dict[str, float]) -> str:
    d = {k: float(positions[k]) for k in sorted(positions)}
    return json.dumps(d, ensure_ascii=False, separators=(",", ":"))


def _portfolio_market_value(
    positions: dict[str, float],
    close_series: pd.Series,
    last_known_price: dict[str, float] | None = None,
) -> float:
    """
    计算持仓组合市值。

    参数:
        positions: 持仓字典，股票代码 -> 持仓股数
        close_series: 当期收盘价序列，索引为股票代码
        last_known_price: 历史最近已知价格字典，用于停牌股票估值；
            当持仓股票不在 close_series 中时，使用该字典中的价格估值

    返回:
        持仓组合总市值（浮点数）
    """
    lkp = last_known_price or {}
    s = 0.0
    for code, sh in positions.items():
        if sh <= 0:
            continue
        if code in close_series.index:
            s += float(sh) * float(close_series.loc[code])
        elif code in lkp:
            s += float(sh) * float(lkp[code])
    return s


def _stuck_non_target_value(
    positions: dict[str, float],
    target: list[str],
    close_series: pd.Series,
) -> float:
    """因跌停等未能卖出的非目标持仓市值（仍计入总权益，但不参与 Top N 等额分配）。"""
    tset = set(target)
    s = 0.0
    for code, sh in positions.items():
        if code in tset:
            continue
        if code in close_series.index and sh > 0:
            s += float(sh) * float(close_series.loc[code])
    return s


def _synthetic_close_from_weekly_returns(df: pd.DataFrame) -> pd.Series:
    """
    无 close 列时，用 future_return_1w 在股票内递推周收盘价（首周锚定 100），仅用于回测与涨跌停价。
    与 split_by_time 等生成的 parquet 兼容。
    """
    idx = df.index
    work = df.sort_values(["stock_code", "date"])
    pieces: list[pd.Series] = []
    for _, g in work.groupby("stock_code", sort=False):
        r = pd.to_numeric(g["future_return_1w"], errors="coerce").to_numpy(dtype=np.float64)
        fac = np.ones(len(r), dtype=np.float64)
        for j in range(1, len(r)):
            x = r[j - 1]
            fac[j] = fac[j - 1] * (1.0 if np.isnan(x) else (1.0 + x))
        pieces.append(pd.Series(100.0 * fac, index=g.index))
    s = pd.concat(pieces)
    return s.reindex(idx)


def _engine_params_snapshot(eng: BacktestEngine, *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    base: dict[str, Any] = {
        "model_type": eng.model_type,
        "top_n": eng.top_n,
        "initial_capital": eng.initial_capital,
        "commission": eng.commission,
        "slippage": eng.slippage,
        "stamp_tax": eng.stamp_tax,
        "vol_penalty": eng.vol_penalty,
        "rebalance_freq": eng._rebalance_freq,
    }
    if extra:
        base.update(extra)
    return base


def _nav_points_from_result_df(result_df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in result_df.iterrows():
        rows.append(
            {
                "date": pd.Timestamp(row["date"]).date().isoformat(),
                "nav": float(row["total_value"]),
            }
        )
    return rows


def _holdings_series_from_result_df(result_df: pd.DataFrame) -> list[dict[str, Any]]:
    """每周持仓明细，与导出的 *_holdings.json 中 series 结构一致。"""
    has_holdings = "holdings" in result_df.columns
    series: list[dict[str, Any]] = []
    for _, row in result_df.iterrows():
        raw = row["holdings"] if has_holdings else None
        holdings = _parse_holdings_cell(raw)
        series.append({"date": pd.Timestamp(row["date"]).date().isoformat(), "holdings": holdings})
    return series


def _parse_holdings_cell(raw: object) -> dict[str, float]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return {}
    if isinstance(raw, str) and raw.strip():
        try:
            return {str(k): float(v) for k, v in json.loads(raw).items()}
        except json.JSONDecodeError:
            return {}
    if isinstance(raw, dict):
        return {str(k): float(v) for k, v in raw.items()}
    return {}


def _price_map_at_date(weekly_df: pd.DataFrame, dt: pd.Timestamp) -> dict[str, float]:
    sub = weekly_df[weekly_df["date"] == dt]
    if sub.empty:
        return {}
    out: dict[str, float] = {}
    for _, r in sub.iterrows():
        c = float(pd.to_numeric(r["close"], errors="coerce") or 0.0)
        if c > 0:
            out[str(r["stock_code"])] = c
    return out


def build_rebalance_turnover_trades(result_df: pd.DataFrame, weekly_df: pd.DataFrame) -> pd.DataFrame:
    """
    由相邻两周持仓差与当周收盘价估算调仓买卖金额，供 turnover_rate 使用（与周频回测口径一致）。
    """
    df = result_df.sort_values("date").reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    prev: dict[str, float] = {}
    for _, row in df.iterrows():
        dt = pd.Timestamp(row["date"])
        curr = _parse_holdings_cell(row["holdings"])
        px = _price_map_at_date(weekly_df, dt)
        buy_amt = 0.0
        sell_amt = 0.0
        for code in set(prev) | set(curr):
            dsh = float(curr.get(code, 0.0) - prev.get(code, 0.0))
            price = px.get(code)
            if price is None or price <= 0 or math.isnan(dsh):
                continue
            if dsh > 0:
                buy_amt += dsh * price
            elif dsh < 0:
                sell_amt += (-dsh) * price
        rows.append(
            {
                "date": dt,
                "buy_amount": buy_amt,
                "sell_amount": sell_amt,
                "total_value": float(row["total_value"]),
            }
        )
        prev = curr
    return pd.DataFrame(rows)


def weekly_portfolio_win_rate(result_df: pd.DataFrame) -> float | None:
    """相邻调仓周组合净值收益 > 0 的周数占比（策略周度胜率）。"""
    if result_df.shape[0] < 2:
        return None
    nav = pd.to_numeric(result_df["total_value"], errors="coerce").astype(float)
    r = nav.pct_change().dropna()
    if r.empty:
        return None
    return float((r > 0).mean())


def average_holding_weeks(result_df: pd.DataFrame) -> float | None:
    """每只股票连续持有周数的简单算术平均（跨所有持有片段）。"""
    if result_df.empty:
        return None
    sets: list[set[str]] = []
    for _, row in result_df.iterrows():
        sets.append(set(_parse_holdings_cell(row["holdings"]).keys()))
    lengths: list[int] = []
    all_codes: set[str] = set()
    for s in sets:
        all_codes |= s
    for code in all_codes:
        in_run = False
        run_len = 0
        for s in sets:
            if code in s:
                if not in_run:
                    in_run = True
                    run_len = 1
                else:
                    run_len += 1
            else:
                if in_run:
                    lengths.append(run_len)
                    in_run = False
                    run_len = 0
        if in_run:
            lengths.append(run_len)
    if not lengths:
        return None
    return float(np.mean(lengths))


def monthly_returns_heatmap_data(dates: list[str], nav_values: list[float]) -> dict[str, Any]:
    """自然月维度：取每月最后一个调仓日净值，计算月度收益率，供热力图矩阵使用。"""
    if len(dates) < 2:
        return {
            "years": [],
            "month_columns": list(range(1, 13)),
            "values": [],
            "note": "样本不足",
        }
    df = pd.DataFrame({"dt": pd.to_datetime(pd.Series(dates)), "nav": nav_values}).sort_values("dt")
    df["ym"] = df["dt"].dt.to_period("M")
    last_nav = df.groupby("ym", sort=True)["nav"].last()
    rets = last_nav.pct_change()
    years = sorted({int(p.year) for p in last_nav.index})
    mat: list[list[float | None]] = []
    for y in years:
        row: list[float | None] = []
        for m in range(1, 13):
            p = pd.Period(year=y, month=m, freq="M")
            if p not in rets.index:
                row.append(None)
                continue
            v = float(rets.loc[p])
            row.append(None if math.isnan(v) else v)
        mat.append(row)
    return {
        "years": years,
        "month_columns": list(range(1, 13)),
        "values": mat,
        "note": "按自然月内最后一个调仓日净值计算月度收益；首月无前值则为空",
    }


def align_comparison_nav_curves(
    result_df_lgb: pd.DataFrame,
    result_df_xgb: pd.DataFrame,
) -> dict[str, Any]:
    """
    两模型净值曲线时间对齐（内连接 date），归一化净值与 LightGBM 相对 XGBoost 的超额（比值为 LGB/XGB-1）。
    """
    if result_df_lgb.empty or result_df_xgb.empty:
        return {
            "granularity": "weekly",
            "dates": [],
            "lightgbm_nav": [],
            "xgboost_nav": [],
            "lightgbm_nav_norm": [],
            "xgboost_nav_norm": [],
            "excess_lightgbm_over_xgb_nav": [],
            "note": "空结果",
        }
    a = result_df_lgb[["date", "total_value"]].rename(columns={"total_value": "nav_lgb"})
    b = result_df_xgb[["date", "total_value"]].rename(columns={"total_value": "nav_xgb"})
    a["date"] = pd.to_datetime(a["date"])
    b["date"] = pd.to_datetime(b["date"])
    m = pd.merge(a, b, on="date", how="inner")
    if m.empty:
        return {
            "granularity": "weekly",
            "dates": [],
            "lightgbm_nav": [],
            "xgboost_nav": [],
            "lightgbm_nav_norm": [],
            "xgboost_nav_norm": [],
            "excess_lightgbm_over_xgb_nav": [],
            "note": "日期交集为空",
        }
    if len(m) != len(a) or len(m) != len(b):
        logger.warning(
            "LightGBM 与 XGBoost 回测调仓日不完全一致，已用交集 %d 条对齐曲线",
            len(m),
        )
    dates = [pd.Timestamp(x).date().isoformat() for x in m["date"]]
    v1 = [float(x) for x in m["nav_lgb"]]
    v2 = [float(x) for x in m["nav_xgb"]]
    if v1[0] <= 0 or v2[0] <= 0:
        n1, n2 = v1, v2
        excess = [0.0] * len(v1)
    else:
        n1 = [x / v1[0] for x in v1]
        n2 = [x / v2[0] for x in v2]
        excess = [a / b - 1.0 if b != 0 else 0.0 for a, b in zip(n1, n2)]
    return {
        "granularity": "weekly",
        "dates": dates,
        "lightgbm_nav": v1,
        "xgboost_nav": v2,
        "lightgbm_nav_norm": n1,
        "xgboost_nav_norm": n2,
        "excess_lightgbm_over_xgb_nav": excess,
        "note": "excess = lightgbm_nav_norm / xgboost_nav_norm - 1",
    }


def metrics_table_with_difference(
    metrics_lightgbm: dict[str, Any],
    metrics_xgboost: dict[str, Any],
) -> list[dict[str, Any]]:
    keys = sorted(set(metrics_lightgbm) | set(metrics_xgboost))
    rows: list[dict[str, Any]] = []
    for k in keys:
        a, b = metrics_lightgbm.get(k), metrics_xgboost.get(k)
        diff: float | None = None
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if not math.isnan(float(a)) and not math.isnan(float(b)):
                diff = float(a) - float(b)
        rows.append({"metric": k, "lightgbm": a, "xgboost": b, "difference": diff})
    return rows


def _scrub_for_json_obj(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _scrub_for_json_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_scrub_for_json_obj(x) for x in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    return obj


def _dual_backtest_core(
    *,
    top_n: int = 10,
    initial_capital: float = 1_000_000.0,
    use_split: Literal["test", "all"] = "test",
    train_end: str = "2020-12-31",
    val_end: str = "2022-12-31",
    skip_initial_weeks: int = 0,
    vol_penalty: float = 0.0,
    data_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """
    加载同一周频数据，用相同参数跑 LightGBM / XGBoost（保证测试区间与调仓日一致）。
    """
    extra_params: dict[str, Any] = {
        "use_split": use_split,
        "train_end": train_end,
        "val_end": val_end,
        "skip_initial_weeks": skip_initial_weeks,
    }
    first_eng = BacktestEngine("lightgbm", top_n, initial_capital, vol_penalty=vol_penalty, data_dir=data_dir)
    weekly_df = first_eng.load_weekly_data()
    packs: dict[str, dict[str, Any]] = {}
    for kind in ("lightgbm", "xgboost"):
        eng = BacktestEngine(kind, top_n, initial_capital, vol_penalty=vol_penalty, data_dir=data_dir)
        result_df = eng.run_backtest(
            weekly_df,
            use_split=use_split,
            train_end=train_end,
            val_end=val_end,
            skip_initial_weeks=skip_initial_weeks,
        )
        params = _engine_params_snapshot(eng, extra=extra_params)
        packs[kind] = {"result_df": result_df, "params": params}
    return weekly_df, packs


def compute_backtest_metrics(
    result_df: pd.DataFrame,
    weekly_df: pd.DataFrame,
    *,
    extended: bool = False,
) -> dict[str, Any]:
    """日频填充后的年化/夏普/回撤；extended 时追加换手率、周度胜率、平均持有周数。"""
    if result_df.empty:
        return {}
    nav_w = pd.Series(result_df["total_value"].to_numpy(), index=pd.to_datetime(result_df["date"]))
    nav_d = weekly_nav_to_daily_business_ffill(nav_w)
    if not extended:
        return aggregate_metrics(nav_d)
    trades = build_rebalance_turnover_trades(result_df, weekly_df)
    m = aggregate_metrics(nav_d, trades_df=trades)
    wr = weekly_portfolio_win_rate(result_df)
    m["win_rate"] = wr
    ah = average_holding_weeks(result_df)
    m["avg_holding_weeks"] = ah
    # Bootstrap置信区间
    try:
        weekly_returns = result_df["weekly_return"].dropna().to_numpy(dtype=np.float64) if "weekly_return" in result_df.columns else np.array([], dtype=np.float64)
        if weekly_returns.size >= 10:
            from .metrics import bootstrap_metric, sharpe_ratio as _sharpe_fn, annualized_return as _ar_fn
            m["sharpe_ci"] = bootstrap_metric(
                weekly_returns,
                lambda r: _sharpe_fn(pd.Series(r, index=pd.date_range("2020-01-01", periods=len(r), freq="B")), risk_free_rate=0.03),
                n_bootstrap=1000,
                confidence=0.95,
            )
            m["annualized_return_ci"] = bootstrap_metric(
                weekly_returns,
                lambda r: _ar_fn(pd.Series(r, index=pd.date_range("2020-01-01", periods=len(r), freq="B"))),
                n_bootstrap=1000,
                confidence=0.95,
            )
    except Exception as exc:
        logger.warning("Bootstrap置信区间计算失败: %s", exc)
    return json.loads(json.dumps(_scrub_for_json_obj(m), ensure_ascii=False, default=str))


def fetch_csi300_benchmark(start: pd.Timestamp, end: pd.Timestamp) -> dict[str, Any]:
    """
    使用 akshare 拉取沪深300（000300）日线收盘，净值 = 收盘价 / 区间首日收盘价。
    内置重试机制和本地缓存，网络失败时返回空序列，source=unavailable。
    """
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    dates: list[str] = []
    navs: list[float] = []
    source = "akshare"
    cache_path = DATA_DIR / "benchmark_cache" / f"csi300_daily_{start_ts.strftime('%Y%m%d')}_{end_ts.strftime('%Y%m%d')}.parquet"
    if cache_path.exists():
        try:
            cached = pd.read_parquet(cache_path)
            dates = cached["date"].tolist()
            navs = cached["nav"].astype(float).tolist()
            source = "cache"
            logger.info("从本地缓存加载沪深300基准净值")
        except Exception:
            pass
    if not dates:
        import time as _time
        for attempt in range(1, 4):
            try:
                import akshare as ak

                df = ak.index_zh_a_hist(
                    symbol="000300",
                    period="daily",
                    start_date=start_ts.strftime("%Y%m%d"),
                    end_date=end_ts.strftime("%Y%m%d"),
                )
                if df is None or df.empty:
                    raise ValueError("empty benchmark dataframe")
                dcol = "日期" if "日期" in df.columns else "date"
                ccol = "收盘" if "收盘" in df.columns else ("close" if "close" in df.columns else None)
                if ccol is None:
                    raise ValueError("benchmark missing close column")
                sub = (
                    pd.DataFrame(
                        {
                            "dt": pd.to_datetime(df[dcol]),
                            "close": pd.to_numeric(df[ccol], errors="coerce"),
                        }
                    )
                    .dropna()
                    .sort_values("dt")
                )
                if sub.empty:
                    raise ValueError("no valid benchmark rows")
                c0 = float(sub["close"].iloc[0])
                if c0 <= 0:
                    raise ValueError("invalid first close")
                dates = [pd.Timestamp(t).date().isoformat() for t in sub["dt"]]
                navs = [float(c / c0) for c in sub["close"]]
                try:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_df = pd.DataFrame({"date": dates, "nav": navs})
                    cache_df.to_parquet(cache_path, index=False)
                    logger.info("沪深300基准净值已缓存")
                except Exception:
                    pass
                break
            except Exception as exc:
                logger.warning("沪深300 基准拉取失败（第%d次尝试）: %s", attempt, exc)
                if attempt < 3:
                    _time.sleep(5.0)
        if not dates:
            source = "unavailable"
    return {
        "index_code": "000300.SH",
        "name": "沪深300",
        "granularity": "daily",
        "dates": dates,
        "nav_values": navs,
        "source": source,
        "range": {
            "start": start_ts.date().isoformat(),
            "end": end_ts.date().isoformat(),
        },
        "note": "nav 以区间内首个有效交易日收盘价归一为 1.0",
    }


def persist_backtest_result(
    *,
    model_type: str,
    params: dict[str, Any],
    result_df: pd.DataFrame,
    metrics: dict[str, Any],
) -> int:
    """写入 backtest_result 表；返回自增 id。"""
    init_db()
    nav_payload = {
        "granularity": "weekly",
        "nav_points": _nav_points_from_result_df(result_df),
        "holdings_series": _holdings_series_from_result_df(result_df),
        "note": "调仓日净值与持仓；指标基于工作日前向填充后的日频序列计算",
    }
    rec = BacktestResult(
        model_type=model_type,
        params_json=json.dumps(params, ensure_ascii=False, default=str),
        nav_json=json.dumps(nav_payload, ensure_ascii=False, default=str),
        metrics_json=metrics_to_json(metrics),
    )
    with session_scope() as session:
        session.add(rec)
        session.flush()
        rid = int(rec.id)
    logger.info("已写入 backtest_result id=%s model=%s", rid, model_type)
    return rid


def build_comparison_json(metrics_lightgbm: dict[str, Any], metrics_xgboost: dict[str, Any]) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "metrics_table": metrics_table_with_difference(metrics_lightgbm, metrics_xgboost),
    }


def save_results_to_json(
    *,
    out_dir: Path,
    lightgbm_pack: dict[str, Any],
    xgboost_pack: dict[str, Any],
    benchmark: dict[str, Any],
) -> None:
    """
    导出前端静态 JSON：各模型 nav / benchmark / metrics / holdings，以及 comparison.json。
    benchmark 内容按模型各写一份（文件名区分），便于前端单模型目录部署。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _write_nav(prefix: str, pack: dict[str, Any]) -> None:
        df = pack["result_df"]
        if df.empty:
            return
        obj = {
            "model_type": prefix,
            "granularity": "weekly",
            "dates": [pd.Timestamp(d).date().isoformat() for d in df["date"]],
            "nav_values": [float(x) for x in df["total_value"]],
        }
        (out_dir / f"{prefix}_nav.json").write_text(
            json.dumps(obj, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _write_metrics(prefix: str, pack: dict[str, Any]) -> None:
        if pack["result_df"].empty:
            return
        (out_dir / f"{prefix}_metrics.json").write_text(
            json.dumps(pack["metrics"], ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    def _write_holdings(prefix: str, pack: dict[str, Any]) -> None:
        df = pack["result_df"]
        if df.empty:
            return
        series = _holdings_series_from_result_df(df)
        obj = {"model_type": prefix, "granularity": "weekly", "series": series}
        (out_dir / f"{prefix}_holdings.json").write_text(
            json.dumps(obj, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _write_benchmark(prefix: str) -> None:
        (out_dir / f"{prefix}_benchmark.json").write_text(
            json.dumps(benchmark, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    for prefix, pack in (("lightgbm", lightgbm_pack), ("xgboost", xgboost_pack)):
        _write_nav(prefix, pack)
        _write_metrics(prefix, pack)
        _write_holdings(prefix, pack)
        _write_benchmark(prefix)

    comp = build_comparison_json(lightgbm_pack.get("metrics", {}), xgboost_pack.get("metrics", {}))
    (out_dir / "comparison.json").write_text(
        json.dumps(comp, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("已导出 JSON 至 %s", out_dir.resolve())


def run_backtest_and_export(
    *,
    top_n: int = 10,
    initial_capital: float = 1_000_000.0,
    use_split: Literal["test", "all"] = "test",
    train_end: str = "2020-12-31",
    val_end: str = "2022-12-31",
    skip_initial_weeks: int = 0,
    vol_penalty: float = 0.0,
    out_dir: Path | None = None,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    """
    对 LightGBM / XGBoost 各跑一次回测，写 SQLite，再写入 data/backtest_results/ 下 JSON。
    返回 {"lightgbm": pack, "xgboost": pack, "benchmark": dict, "out_dir": str}
    """
    out = Path(out_dir) if out_dir is not None else BACKTEST_RESULTS_DIR
    weekly_df, packs = _dual_backtest_core(
        top_n=top_n,
        initial_capital=initial_capital,
        use_split=use_split,
        train_end=train_end,
        val_end=val_end,
        skip_initial_weeks=skip_initial_weeks,
        vol_penalty=vol_penalty,
        data_dir=data_dir,
    )

    for kind in ("lightgbm", "xgboost"):
        result_df = packs[kind]["result_df"]
        params = packs[kind]["params"]
        if result_df.empty:
            logger.warning("模型 %s 回测结果为空，跳过落库与部分 JSON", kind)
            packs[kind]["metrics"] = {}
            continue
        metrics = compute_backtest_metrics(result_df, weekly_df, extended=False)
        persist_backtest_result(model_type=kind, params=params, result_df=result_df, metrics=metrics)
        packs[kind]["metrics"] = metrics

    all_dates: list[pd.Timestamp] = []
    for p in packs.values():
        df = p["result_df"]
        if not df.empty:
            all_dates.extend(pd.to_datetime(df["date"]).tolist())
    if all_dates:
        bench = fetch_csi300_benchmark(pd.Timestamp(min(all_dates)), pd.Timestamp(max(all_dates)))
    else:
        bench = fetch_csi300_benchmark(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02"))

    save_results_to_json(
        out_dir=out,
        lightgbm_pack=packs["lightgbm"],
        xgboost_pack=packs["xgboost"],
        benchmark=bench,
    )

    return {
        "lightgbm": packs["lightgbm"],
        "xgboost": packs["xgboost"],
        "benchmark": bench,
        "out_dir": str(out.resolve()),
    }


def write_backtest_comparison_html(nav_pack: dict[str, Any], out_path: Path) -> bool:
    """Plotly 双净值 + 超额曲线；未安装 plotly 时返回 False。"""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        logger.warning("未安装 plotly，跳过 HTML 报告（pip install plotly）")
        return False
    dates = nav_pack.get("dates") or []
    if not dates:
        logger.warning("无对齐净值数据，跳过 HTML")
        return False
    n1 = nav_pack["lightgbm_nav_norm"]
    n2 = nav_pack["xgboost_nav_norm"]
    ex = nav_pack["excess_lightgbm_over_xgb_nav"]
    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.62, 0.38],
        vertical_spacing=0.1,
        subplot_titles=("归一化净值（起点=1）", "LightGBM 相对 XGBoost 超额（LGB/XGB - 1）"),
    )
    fig.add_trace(
        go.Scatter(x=dates, y=n1, name="LightGBM", line=dict(width=1.5)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=dates, y=n2, name="XGBoost", line=dict(width=1.5)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=dates, y=ex, name="超额", marker_color="rgba(80,80,120,0.55)"),
        row=2,
        col=1,
    )
    fig.update_layout(
        height=720,
        title_text="LightGBM vs XGBoost 策略对比",
        showlegend=True,
        template="plotly_white",
    )
    fig.update_xaxes(title_text="日期", row=2, col=1)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn", full_html=True)
    logger.info("已写入 Plotly 报告 %s", out_path.resolve())
    return True


def run_comparison(
    *,
    top_n: int = 10,
    initial_capital: float = 1_000_000.0,
    use_split: Literal["test", "all"] = "test",
    train_end: str = "2020-12-31",
    val_end: str = "2022-12-31",
    skip_initial_weeks: int = 0,
    vol_penalty: float = 0.0,
    out_dir: Path | None = None,
    data_dir: Path | None = None,
    write_html: bool = True,
    html_path: Path | None = None,
) -> dict[str, Any]:
    """
    独立对比：同一数据、同一参数跑两模型；扩展指标（换手、胜率、平均持有周数）；
    写 comparison.json（含热力图与并排表）、comparison_nav.json；可选 Plotly HTML。
    """
    out = Path(out_dir) if out_dir is not None else BACKTEST_RESULTS_DIR
    weekly_df, packs = _dual_backtest_core(
        top_n=top_n,
        initial_capital=initial_capital,
        use_split=use_split,
        train_end=train_end,
        val_end=val_end,
        skip_initial_weeks=skip_initial_weeks,
        vol_penalty=vol_penalty,
        data_dir=data_dir,
    )

    for kind in ("lightgbm", "xgboost"):
        result_df = packs[kind]["result_df"]
        if result_df.empty:
            packs[kind]["metrics"] = {}
            continue
        packs[kind]["metrics"] = compute_backtest_metrics(result_df, weekly_df, extended=True)

    m_lgb = packs["lightgbm"].get("metrics") or {}
    m_xgb = packs["xgboost"].get("metrics") or {}
    dfl = packs["lightgbm"]["result_df"]
    dfx = packs["xgboost"]["result_df"]

    nav_pack = align_comparison_nav_curves(dfl, dfx)
    heat_lgb = monthly_returns_heatmap_data(
        [pd.Timestamp(x).date().isoformat() for x in dfl["date"]] if not dfl.empty else [],
        [float(x) for x in dfl["total_value"]] if not dfl.empty else [],
    )
    heat_xgb = monthly_returns_heatmap_data(
        [pd.Timestamp(x).date().isoformat() for x in dfx["date"]] if not dfx.empty else [],
        [float(x) for x in dfx["total_value"]] if not dfx.empty else [],
    )

    comparison_doc: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "kind": "strategy_comparison",
        "params_common": packs["lightgbm"].get("params") or packs["xgboost"].get("params"),
        "metrics_lightgbm": m_lgb,
        "metrics_xgboost": m_xgb,
        "metrics_table": metrics_table_with_difference(m_lgb, m_xgb),
        "monthly_returns_heatmap": {
            "lightgbm": heat_lgb,
            "xgboost": heat_xgb,
        },
        "comparison_nav_file": "comparison_nav.json",
        "notes": {
            "nav_alignment": nav_pack.get("note"),
            "win_rate": "周度胜率：相邻调仓周组合净值收益为正的周占比",
            "avg_holding_weeks": "平均持有周数：各标的连续持有周数的算术平均",
        },
    }
    comparison_doc = _scrub_for_json_obj(comparison_doc)

    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.json").write_text(
        json.dumps(comparison_doc, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    nav_out = _scrub_for_json_obj(nav_pack)
    (out / "comparison_nav.json").write_text(
        json.dumps(nav_out, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("已写入对比 JSON：%s", (out / "comparison.json").resolve())

    html_written = False
    html_resolved: str | None = None
    if write_html:
        hp = Path(html_path) if html_path is not None else REPORTS_DIR / "backtest_comparison.html"
        html_written = write_backtest_comparison_html(nav_pack, hp)
        if html_written:
            html_resolved = str(hp.resolve())

    return {
        "lightgbm": packs["lightgbm"],
        "xgboost": packs["xgboost"],
        "comparison": comparison_doc,
        "comparison_nav": nav_pack,
        "out_dir": str(out.resolve()),
        "html_report": html_resolved,
    }


def run_multi_topn_backtest(
    top_n_list: list[int] | None = None,
    model_type: ModelKind = "lightgbm",
    initial_capital: float = 1_000_000.0,
    *,
    use_split: Literal["test", "all"] = "test",
    train_end: str = "2020-12-31",
    val_end: str = "2022-12-31",
    skip_initial_weeks: int = 0,
    vol_penalty: float = 0.0,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    """
    多Top N参数对比回测：遍历多个选股数量，汇总指标对比表。

    参数:
        top_n_list: 选股数量列表，默认 [5, 10, 20, 30]
        model_type: 模型类型
        initial_capital: 初始资金
        use_split: 数据切分方式
        train_end: 训练集截止日期
        val_end: 验证集截止日期
        skip_initial_weeks: 跳过初始周数
        vol_penalty: 波动率惩罚系数，默认 0.0（不调整）
        data_dir: 数据目录

    返回:
        {"results": {top_n: result_df}, "metrics_table": DataFrame, "summary": dict}
        metrics_table 为各Top N的核心指标汇总表
    """
    if top_n_list is None:
        top_n_list = [5, 10, 20, 30]

    first_eng = BacktestEngine(model_type, top_n_list[0], initial_capital, vol_penalty=vol_penalty, data_dir=data_dir)
    weekly_df = first_eng.load_weekly_data()

    all_metrics: dict[int, dict[str, Any]] = {}
    all_results: dict[int, pd.DataFrame] = {}

    for tn in top_n_list:
        eng = BacktestEngine(model_type, tn, initial_capital, vol_penalty=vol_penalty, data_dir=data_dir)
        result_df = eng.run_backtest(
            weekly_df,
            use_split=use_split,
            train_end=train_end,
            val_end=val_end,
            skip_initial_weeks=skip_initial_weeks,
        )
        all_results[tn] = result_df
        if result_df.empty:
            all_metrics[tn] = {}
            continue
        metrics = compute_backtest_metrics(result_df, weekly_df, extended=False)
        all_metrics[tn] = metrics

    rows: list[dict[str, Any]] = []
    for tn in top_n_list:
        m = all_metrics.get(tn, {})
        row: dict[str, Any] = {"top_n": tn}
        for k in ["annualized_return", "sharpe_ratio", "max_drawdown", "calmar_ratio", "sortino_ratio"]:
            row[k] = m.get(k)
        rows.append(row)

    metrics_table = pd.DataFrame(rows) if rows else pd.DataFrame()

    summary: dict[str, Any] = {
        "model_type": model_type,
        "top_n_list": top_n_list,
        "initial_capital": initial_capital,
        "use_split": use_split,
    }

    return {
        "results": all_results,
        "metrics_table": metrics_table,
        "summary": summary,
    }


def run_random_baseline(
    top_n: int = 20,
    initial_capital: float = 1_000_000.0,
    n_runs: int = 100,
    seed: int = 42,
    *,
    use_split: Literal["test", "all"] = "test",
    train_end: str = "2020-12-31",
    val_end: str = "2022-12-31",
    skip_initial_weeks: int = 0,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    """
    B-EW等权随机选股基线：每期从截面中随机选N只股票等权配置，跑多次取平均。

    参数:
        top_n: 每期选股数量
        initial_capital: 初始资金
        n_runs: 随机运行次数
        seed: 随机种子
        use_split: 数据切分方式
        train_end: 训练集截止日期
        val_end: 验证集截止日期
        skip_initial_weeks: 跳过初始周数
        data_dir: 数据目录

    返回:
        {"avg_nav": 平均净值序列, "avg_metrics": 平均指标, "all_navs": 所有净值序列, "all_metrics": 所有指标}
    """
    rng = np.random.default_rng(seed)
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent / "data"

    weekly_df = BacktestEngine("lightgbm", top_n, initial_capital, data_dir=data_dir).load_weekly_data()
    if use_split == "test":
        cutoff = pd.Timestamp(val_end)
        weekly_df = weekly_df[weekly_df["date"] > cutoff].copy()

    dates = sorted(weekly_df["date"].unique())

    all_navs: list[np.ndarray] = []
    all_metrics: list[dict[str, Any]] = []

    for run_i in range(n_runs):
        run_rng = np.random.default_rng(rng.integers(0, 2**31))
        nav_points: list[float] = [1.0]
        cash = initial_capital
        positions: dict[str, dict[str, Any]] = {}

        for i, d in enumerate(dates):
            section = weekly_df[weekly_df["date"] == d]
            available = section["stock_code"].unique().tolist()
            n_pick = min(top_n, len(available))
            chosen = run_rng.choice(available, size=n_pick, replace=False).tolist()

            px = section.drop_duplicates("stock_code").set_index("stock_code")["close"] if "close" in section.columns else pd.Series(dtype=float)

            # 卖出非目标
            sell_proceeds = 0.0
            for code in list(positions.keys()):
                if code not in chosen:
                    if code in px.index:
                        sell_price = float(px.loc[code]) * (1 - 0.001)
                        sell_proceeds += positions[code]["shares"] * sell_price
                        sell_proceeds -= positions[code]["shares"] * sell_price * 0.0003
                        sell_proceeds -= positions[code]["shares"] * sell_price * 0.0005
                    del positions[code]

            cash += sell_proceeds

            # 买入目标
            buy_codes = [c for c in chosen if c not in positions]
            if buy_codes:
                total_value = cash + sum(
                    positions[c]["shares"] * float(px.loc[c]) if c in px.index else 0.0
                    for c in positions
                )
                per_stock = total_value / max(len(chosen), 1)
                for code in buy_codes:
                    if code in px.index:
                        buy_price = float(px.loc[code]) * (1 + 0.001)
                        shares = per_stock / buy_price if buy_price > 0 else 0.0
                        cost = shares * buy_price * (1 + 0.0003)
                        cash -= cost
                        positions[code] = {"shares": shares, "price": buy_price}

            # 计算总权益
            total = cash + sum(
                positions[c]["shares"] * float(px.loc[c]) if c in px.index else 0.0
                for c in positions
            )
            nav_points.append(total / initial_capital)

        nav_arr = np.array(nav_points)
        all_navs.append(nav_arr)

        # 计算指标
        rets = np.diff(nav_arr)
        if len(rets) > 1:
            ann_ret = float(np.mean(rets) * 52)
            ann_vol = float(np.std(rets, ddof=1) * np.sqrt(52))
            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
            cummax = np.maximum.accumulate(nav_arr)
            dd = (nav_arr - cummax) / cummax
            mdd = float(np.min(dd))
        else:
            ann_ret = sharpe = mdd = 0.0
        all_metrics.append({"annualized_return": ann_ret, "sharpe_ratio": sharpe, "max_drawdown": abs(mdd)})

    # 取平均
    max_len = max(len(n) for n in all_navs)
    padded = np.array([np.pad(n, (0, max_len - len(n)), constant_values=n[-1]) for n in all_navs])
    avg_nav = np.mean(padded, axis=0)

    avg_metrics: dict[str, float] = {}
    for key in all_metrics[0]:
        avg_metrics[key] = float(np.mean([m[key] for m in all_metrics]))

    return {"avg_nav": avg_nav, "avg_metrics": avg_metrics, "all_navs": all_navs, "all_metrics": all_metrics}


def run_momentum_baseline(
    top_n: int = 20,
    initial_capital: float = 1_000_000.0,
    *,
    momentum_col: str = "mom_6m",
    use_split: Literal["test", "all"] = "test",
    train_end: str = "2020-12-31",
    val_end: str = "2022-12-31",
    skip_initial_weeks: int = 0,
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """
    B-MOM纯动量排序基线：按动量因子降序排列选股，无ML模型。

    参数:
        top_n: 每期选股数量
        initial_capital: 初始资金
        momentum_col: 动量因子列名
        use_split: 数据切分方式
        train_end: 训练集截止日期
        val_end: 验证集截止日期
        skip_initial_weeks: 跳过初始周数
        data_dir: 数据目录

    返回:
        回测结果DataFrame（与run_backtest格式一致）
    """
    from .data_loader import load_factor_columns, fill_missing_factors

    if data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent / "data"

    weekly_df = BacktestEngine("lightgbm", top_n, initial_capital, data_dir=data_dir).load_weekly_data()
    if use_split == "test":
        cutoff = pd.Timestamp(val_end)
        weekly_df = weekly_df[weekly_df["date"] > cutoff].copy()

    factor_cols = load_factor_columns(data_dir=data_dir)
    dates = sorted(weekly_df["date"].unique())

    cash = initial_capital
    positions: dict[str, dict[str, Any]] = {}
    last_known_price: dict[str, float] = {}
    records: list[dict[str, Any]] = []

    for i, d in enumerate(dates):
        section = weekly_df[weekly_df["date"] == d].copy()

        # 用动量因子排序
        if momentum_col in section.columns:
            section = section.dropna(subset=[momentum_col])
            section = section.sort_values(momentum_col, ascending=False)
        else:
            logger.warning("截面 %s 中无动量因子列 %s，跳过", d, momentum_col)
            continue

        target_codes = section["stock_code"].head(top_n).tolist()

        px = section.drop_duplicates("stock_code").set_index("stock_code")
        close_col = "close" if "close" in px.columns else px.columns[0]

        # 卖出非目标
        sell_proceeds = 0.0
        for code in list(positions.keys()):
            if code not in target_codes:
                if code in px.index:
                    sell_price = float(px.loc[code, close_col]) * (1 - 0.001)
                    amount = positions[code]["shares"] * sell_price
                    sell_proceeds += amount
                    sell_proceeds -= amount * 0.0003
                    sell_proceeds -= amount * 0.0005
                del positions[code]
        cash += sell_proceeds

        # 买入目标
        buy_codes = [c for c in target_codes if c not in positions and c in px.index]
        if buy_codes:
            mv = cash + sum(
                positions[c]["shares"] * float(px.loc[c, close_col])
                for c in positions if c in px.index
            )
            per_stock = mv / max(len(target_codes), 1)
            for code in buy_codes:
                buy_price = float(px.loc[code, close_col]) * (1 + 0.001)
                shares = per_stock / buy_price if buy_price > 0 else 0.0
                cost = shares * buy_price * (1 + 0.0003)
                cash -= cost
                positions[code] = {"shares": shares, "price": buy_price}

        for code in positions:
            if code in px.index:
                last_known_price[code] = float(px.loc[code, close_col])

        total_value = cash + sum(
            positions[c]["shares"] * last_known_price.get(c, positions[c].get("price", 0))
            for c in positions
        )

        records.append({
            "date": d,
            "total_value": total_value,
            "cash": cash,
            "n_positions": len(positions),
            "holdings": list(positions.keys()),
        })

    result = pd.DataFrame(records)
    if not result.empty:
        result["nav"] = result["total_value"] / initial_capital
        result["weekly_return"] = result["total_value"].pct_change(fill_method=None)
        result.loc[result.index[0], "weekly_return"] = 0.0
    return result


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    eng = BacktestEngine("lightgbm", top_n=10, initial_capital=1_000_000.0)
    out = eng.run_backtest(use_split="test")
    print(out.head())
    print("...")
    print(out.tail())
    if not out.empty:
        print("final_nav", float(out["total_value"].iloc[-1]))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="周频回测：快速预览或 --run 持久化+导出 JSON")
    parser.add_argument(
        "--run",
        action="store_true",
        help="运行 LightGBM+XGBoost 回测，写入 SQLite(backtest_result) 与 data/backtest_results/*.json",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="仅跑双模型对比：扩展指标、comparison.json、comparison_nav.json，可选 Plotly HTML",
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="与 --compare 联用：不生成 reports/backtest_comparison.html",
    )
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--initial-capital", type=float, default=1_000_000.0)
    parser.add_argument("--use-split", choices=("test", "all"), default="test")
    parser.add_argument("--train-end", default="2020-12-31")
    parser.add_argument("--val-end", default="2022-12-31")
    parser.add_argument("--skip-initial-weeks", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="", help="JSON 输出目录，默认 backend/data/backtest_results")
    args = parser.parse_args()
    od = Path(args.out_dir) if args.out_dir.strip() else None
    if args.compare:
        run_comparison(
            top_n=args.top_n,
            initial_capital=args.initial_capital,
            use_split=args.use_split,  # type: ignore[arg-type]
            train_end=args.train_end,
            val_end=args.val_end,
            skip_initial_weeks=args.skip_initial_weeks,
            out_dir=od,
            write_html=not args.no_html,
        )
    elif args.run:
        run_backtest_and_export(
            top_n=args.top_n,
            initial_capital=args.initial_capital,
            use_split=args.use_split,  # type: ignore[arg-type]
            train_end=args.train_end,
            val_end=args.val_end,
            skip_initial_weeks=args.skip_initial_weeks,
            out_dir=od,
        )
    else:
        _main()
