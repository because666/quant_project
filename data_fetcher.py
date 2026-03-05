"""
数据获取与处理模块
================

本模块是量化投资选股系统的数据基础模块，负责：
1. 从AKShare获取A股历史数据
2. 计算技术指标（MA、EMA、MACD、RSI、布林带等）
3. 构建收益率特征和目标变量
4. 数据预处理和标准化

主要类：
- StockDataFetcher: 股票数据获取类
- FeatureEngineer: 特征工程类
- DataPreprocessor: 数据预处理类

作者：量化投资团队
日期：2026年1月
"""

import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
from tqdm import tqdm
import ta
from typing import List, Dict, Any, Optional, Union, Tuple
import sqlite3
import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockDataFetcher:
    """
    股票数据获取类
    
    该类负责从AKShare获取A股历史数据，并支持将数据存储到SQLite数据库和CSV文件。
    
    属性：
        data_dir (Path): 数据存储目录路径
        db_path (Path): SQLite数据库文件路径
    
    使用示例：
        >>> fetcher = StockDataFetcher()
        >>> df = fetcher.fetch_stock_data('600519', '2020-01-01', '2025-12-31')
        >>> fetcher.save_data(df, 'stock_data.csv')
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        初始化数据获取器
        
        参数：
            data_dir (str, optional): 数据存储目录，默认为 'data'
        """
        self.data_dir = Path(data_dir) if data_dir else Path("data")
        self.data_dir.mkdir(exist_ok=True)
        self.db_path = self.data_dir / "stock_data.db"
        self.init_db()

    def get_connection(self):
        """
        获取SQLite数据库连接
        
        返回：
            sqlite3.Connection: 数据库连接对象
        
        注意：
            check_same_thread=False 允许多线程访问数据库
        """
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def init_db(self):
        """
        初始化SQLite数据库
        
        创建数据库文件（如果不存在），为后续数据存储做准备。
        表结构会在save_to_db时根据DataFrame动态创建。
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        conn.close()

    def fetch_stock_data(self, stock_code: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        获取单只股票的历史数据
        
        从AKShare获取指定股票的日线数据，包括开高低收、成交量等。
        
        参数：
            stock_code (str): 股票代码，如 '600519'（贵州茅台）
            start_date (str): 开始日期，格式 'YYYY-MM-DD'
            end_date (str): 结束日期，格式 'YYYY-MM-DD'
        
        返回：
            pd.DataFrame: 股票历史数据，包含以下列：
                - date: 日期
                - stock_code: 股票代码
                - open: 开盘价
                - close: 收盘价
                - high: 最高价
                - low: 最低价
                - volume: 成交量
                - amount: 成交额
                - amplitude: 振幅
                - pct_change: 涨跌幅
                - change: 涨跌额
                - turnover: 换手率
            如果获取失败返回 None
        
        注意：
            - 使用前复权(qfq)方式调整价格
            - 股票代码会自动补齐为6位
        
        示例：
            >>> df = fetcher.fetch_stock_data('600519', '2020-01-01', '2025-12-31')
            >>> print(df.columns.tolist())
        """
        try:
            stock_code_6 = stock_code.zfill(6)
            
            df = ak.stock_zh_a_hist(
                symbol=stock_code_6,
                period="daily",
                start_date=start_date.replace('-', ''),
                end_date=end_date.replace('-', ''),
                adjust="qfq"
            )
            
            if df.empty:
                logger.warning(f"No data found for stock {stock_code}")
                return None
            
            df['stock_code'] = stock_code
            df['date'] = pd.to_datetime(df['日期'])
            df = df.rename(columns={
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume',
                '成交额': 'amount',
                '振幅': 'amplitude',
                '涨跌幅': 'pct_change',
                '涨跌额': 'change',
                '换手率': 'turnover'
            })
            
            df = df.sort_values('date').reset_index(drop=True)
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {stock_code}: {e}")
            return None

    def fetch_fundamental_data(self, stock_code: str) -> Optional[pd.DataFrame]:
        """
        获取股票基本面数据
        
        获取市盈率(PE)、市净率(PB)、总市值等基本面指标。
        
        参数：
            stock_code (str): 股票代码
        
        返回：
            pd.DataFrame: 基本面数据，包含：
                - date: 日期
                - pe_ratio: 市盈率
                - pb_ratio: 市净率
                - total_market_cap: 总市值
            如果获取失败返回 None
        
        注意：
            基本面数据通常更新频率较低，需要与日线数据合并后前向填充
        """
        try:
            stock_code_6 = stock_code.zfill(6)
            
            try:
                if hasattr(ak, 'stock_a_indicator_lg'):
                    df = ak.stock_a_indicator_lg(symbol=stock_code_6)
                else:
                    return None
            except:
                 return None

            if df is None or df.empty:
                return None
                
            df['date'] = pd.to_datetime(df['trade_date'])
            df = df.rename(columns={
                'pe': 'pe_ratio',
                'pb': 'pb_ratio',
                'ps': 'ps_ratio',
                'total_mv': 'total_market_cap'
            })
            cols = ['date', 'pe_ratio', 'pb_ratio', 'total_market_cap']
            cols = [c for c in cols if c in df.columns]
            
            return df[cols].sort_values('date')
            
        except Exception as e:
            logger.warning(f"Error fetching fundamental data for {stock_code}: {e}")
            return None

    def fetch_index_data(self, start_date: str, end_date: str, index_code: str = "000001") -> Optional[pd.DataFrame]:
        """
        获取市场指数数据
        
        获取上证指数等市场基准数据，用于计算市场收益率和基准比较。
        
        参数：
            start_date (str): 开始日期
            end_date (str): 结束日期
            index_code (str): 指数代码，默认 '000001'（上证指数）
                              '000300' 为沪深300指数
        
        返回：
            pd.DataFrame: 指数数据，包含：
                - date: 日期
                - market_return: 市场日收益率
        
        示例：
            >>> index_df = fetcher.fetch_index_data('2020-01-01', '2025-12-31', '000300')
        """
        try:
            df = ak.stock_zh_index_daily_em(symbol="sh" + index_code, start_date=start_date.replace('-', ''), end_date=end_date.replace('-', ''))
            df['date'] = pd.to_datetime(df['date'])
            df = df.rename(columns={'close': 'index_close'})
            
            df['market_return'] = df['index_close'].pct_change()
            
            return df[['date', 'market_return']]
        except Exception as e:
            logger.warning(f"Error fetching index data: {e}")
            return None

    @st.cache_data(ttl=3600*12)
    def fetch_multiple_stocks(_self, stock_codes: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        批量获取多只股票数据
        
        一次性获取多只股票的历史数据，并合并基本面数据和指数数据。
        使用Streamlit缓存机制，避免重复请求。
        
        参数：
            stock_codes (List[str]): 股票代码列表
            start_date (str): 开始日期
            end_date (str): 结束日期
        
        返回：
            pd.DataFrame: 合并后的所有股票数据
        
        注意：
            - _self 参数前加下划线是为了避免被 st.cache_data 哈希
            - 缓存时间为12小时（ttl=3600*12）
        
        示例：
            >>> stock_list = ['600519', '000858', '000333']
            >>> df = fetcher.fetch_multiple_stocks(stock_list, '2020-01-01', '2025-12-31')
        """
        all_data = []
        
        index_df = _self.fetch_index_data(start_date, end_date)
        
        for stock_code in tqdm(stock_codes, desc="Fetching stock data"):
            df = _self.fetch_stock_data(stock_code, start_date, end_date)
            
            if df is not None:
                try:
                    fun_df = _self.fetch_fundamental_data(stock_code)
                    if fun_df is not None and not fun_df.empty:
                        df = pd.merge(df, fun_df, on='date', how='left')
                        fun_cols = ['pe_ratio', 'pb_ratio', 'total_market_cap']
                        valid_fun_cols = [c for c in fun_cols if c in df.columns]
                        if valid_fun_cols:
                            df[valid_fun_cols] = df[valid_fun_cols].ffill()
                except Exception as e:
                    logger.warning(f"Skipping fundamentals for {stock_code}: {e}")

                if index_df is not None:
                    df = pd.merge(df, index_df, on='date', how='left')
                
                all_data.append(df)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            return combined_df
        else:
            return pd.DataFrame()

    def save_data(self, df: pd.DataFrame, filename: str) -> None:
        """
        保存数据到CSV和SQLite数据库
        
        同时保存数据到CSV文件（备份）和SQLite数据库（高性能查询）。
        
        参数：
            df (pd.DataFrame): 要保存的数据
            filename (str): 文件名，如 'stock_data.csv'
        
        注意：
            - CSV文件使用UTF-8-BOM编码，确保中文正确显示
            - SQLite会自动创建 (stock_code, date) 索引加速查询
        """
        filepath = self.data_dir / filename
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        logger.info(f"Data saved to {filepath}")
        
        try:
            conn = self.get_connection()
            table_name = filename.replace('.csv', '').replace('.', '_')
            
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            
            cursor = conn.cursor()
            cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_code_date ON {table_name}(stock_code, date)")
            conn.commit()
            conn.close()
            logger.info(f"Data synced to SQLite table {table_name}")
        except Exception as e:
            logger.error(f"Failed to save to SQLite: {e}")

    def load_data(self, filename: str) -> Optional[pd.DataFrame]:
        """
        从SQLite或CSV加载数据
        
        优先从SQLite加载数据（更快），如果失败则回退到CSV文件。
        
        参数：
            filename (str): 文件名，如 'stock_data.csv'
        
        返回：
            pd.DataFrame: 加载的数据，如果文件不存在返回 None
        """
        table_name = filename.replace('.csv', '').replace('.', '_')
        
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
            if cursor.fetchone():
                df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
                df['date'] = pd.to_datetime(df['date'])
                conn.close()
                logger.info(f"Loaded {len(df)} rows from SQLite table {table_name}")
                return df
            conn.close()
        except Exception as e:
            logger.warning(f"SQLite load failed, falling back to CSV: {e}")
        
        filepath = self.data_dir / filename
        if filepath.exists():
            df = pd.read_csv(filepath, encoding='utf-8-sig')
            df['date'] = pd.to_datetime(df['date'])
            return df
        else:
            logger.warning(f"File {filepath} not found")
            return None


class FeatureEngineer:
    """
    特征工程类
    
    该类负责计算技术指标、构建收益率特征和目标变量。
    是连接原始数据和机器学习模型的桥梁。
    
    主要功能：
    1. 计算技术指标：MA、EMA、MACD、RSI、布林带、ATR、OBV等
    2. 计算收益率特征：不同周期的收益率、波动率、偏度、峰度
    3. 构建目标变量：未来N天的涨跌标签
    
    使用示例：
        >>> engineer = FeatureEngineer()
        >>> df = engineer.add_technical_indicators(raw_df)
        >>> df = engineer.add_return_features(df)
        >>> df = engineer.add_target_variable(df, prediction_days=5)
    """
    
    def __init__(self):
        """初始化特征工程器"""
        pass

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加技术指标特征
        
        计算常用的技术分析指标，包括趋势指标、动量指标、波动率指标等。
        
        参数：
            df (pd.DataFrame): 原始股票数据，必须包含 open/high/low/close/volume 列
        
        返回：
            pd.DataFrame: 添加技术指标后的数据
        
        计算的技术指标：
            趋势指标：
                - ma5, ma10, ma20, ma60: 简单移动平均线
                - ema12, ema26: 指数移动平均线
                - macd, macd_signal, macd_diff: MACD指标
            
            动量指标：
                - rsi: 相对强弱指数（14日）
                - momentum: 动量指标（10日）
                - stoch_k, stoch_d: 随机指标
            
            波动率指标：
                - bb_high, bb_mid, bb_low: 布林带上下轨和中轨
                - bb_width: 布林带宽度
                - atr: 平均真实波幅（14日）
            
            成交量指标：
                - obv: 能量潮指标
                - volume_ma5: 5日成交量均值
                - volume_ratio: 量比
            
            其他：
                - price_position: 当日价格位置（收盘价在最高最低价之间的位置）
        
        注意：
            - 技术指标按股票分组计算，避免不同股票数据混淆
            - 计算产生的NaN值会使用前向填充和后向填充处理
        """
        df = df.copy()
        
        grouped = df.groupby('stock_code')
        processed_groups = []
        
        for stock_code, group in tqdm(grouped, desc="Calculating technical indicators"):
            group = group.copy().sort_values('date').reset_index(drop=True)
            
            close = group['close']
            high = group['high']
            low = group['low']
            volume = group['volume']
            
            group['ma5'] = ta.trend.sma_indicator(close, window=5)
            group['ma10'] = ta.trend.sma_indicator(close, window=10)
            group['ma20'] = ta.trend.sma_indicator(close, window=20)
            group['ma60'] = ta.trend.sma_indicator(close, window=60)
            
            group['ema12'] = ta.trend.ema_indicator(close, window=12)
            group['ema26'] = ta.trend.ema_indicator(close, window=26)
            
            macd = ta.trend.MACD(close)
            group['macd'] = macd.macd()
            group['macd_signal'] = macd.macd_signal()
            group['macd_diff'] = macd.macd_diff()
            
            group['rsi'] = ta.momentum.rsi(close, window=14)
            
            bb = ta.volatility.BollingerBands(close)
            group['bb_high'] = bb.bollinger_hband()
            group['bb_mid'] = bb.bollinger_mavg()
            group['bb_low'] = bb.bollinger_lband()
            group['bb_width'] = group['bb_high'] - group['bb_low']
            group['bollinger_width'] = group['bb_width']
            
            group['atr'] = ta.volatility.average_true_range(high, low, close, window=14)
            
            group['obv'] = ta.volume.on_balance_volume(close, volume)
            
            stoch_rsi = ta.momentum.StochRSIIndicator(close, window=14, smooth1=3, smooth2=3)
            group['stoch_k'] = stoch_rsi.stochrsi_k()
            group['stoch_d'] = stoch_rsi.stochrsi_d()
            
            group['momentum'] = ta.momentum.roc(close, window=10)
            
            group['volume_ma5'] = volume.rolling(window=5).mean()
            group['volume_ratio'] = volume / group['volume_ma5']
            
            group['price_position'] = (close - low) / (high - low + 1e-8)
            
            processed_groups.append(group)
        
        if not processed_groups:
            return df
            
        combined_df = pd.concat(processed_groups, ignore_index=True)
        
        combined_df = combined_df.replace([np.inf, -np.inf], np.nan)
        combined_df = combined_df.sort_values(['stock_code', 'date'])
        
        combined_df_filled = combined_df.groupby('stock_code').ffill().bfill()
        
        if 'stock_code' not in combined_df_filled.columns:
            combined_df_filled['stock_code'] = combined_df['stock_code']
        if 'date' not in combined_df_filled.columns:
            combined_df_filled['date'] = combined_df['date']
            
        return combined_df_filled

    def add_return_features(self, df: pd.DataFrame, lookback_days: List[int] = [1, 3, 5, 10, 20]) -> pd.DataFrame:
        """
        添加收益率特征
        
        计算不同周期的收益率、波动率、偏度、峰度等统计特征。
        
        参数：
            df (pd.DataFrame): 包含收盘价的数据
            lookback_days (List[int]): 回看天数列表，默认 [1, 3, 5, 10, 20]
        
        返回：
            pd.DataFrame: 添加收益率特征后的数据
        
        计算的特征：
            收益率类：
                - return_{n}d: n日收益率
                - return_{n}d_max: n日最大收益率
                - return_{n}d_min: n日最小收益率
            
            波动率类：
                - volatility_5d: 5日波动率（收益率标准差）
                - volatility_10d: 10日波动率
                - volatility_20d: 20日波动率
            
            统计特征：
                - skewness_10d: 10日收益率偏度（衡量分布不对称性）
                - kurtosis_10d: 10日收益率峰度（衡量分布尾部厚度）
        
        注意：
            偏度 > 0 表示右偏（正收益概率更高）
            峰度 > 0 表示厚尾（极端收益概率更高）
        """
        df = df.copy()
        
        if 'stock_code' not in df.columns:
            logger.warning("Column 'stock_code' not found in DataFrame. Return features will be calculated globally (not recommended for multiple stocks).")
            return df

        grouped = df.groupby('stock_code')
        processed_groups = []
        
        for stock_code, group in tqdm(grouped, desc="Calculating return features"):
            group = group.copy().sort_values('date').reset_index(drop=True)
            close = group['close']
            
            for days in lookback_days:
                group[f'return_{days}d'] = close.pct_change(days)
                group[f'return_{days}d_max'] = close.rolling(days).max().pct_change(days)
                group[f'return_{days}d_min'] = close.rolling(days).min().pct_change(days)
            
            return_1d = group['return_1d']
            group['volatility_5d'] = return_1d.rolling(5).std()
            group['volatility_10d'] = return_1d.rolling(10).std()
            group['volatility_20d'] = return_1d.rolling(20).std()
            
            group['skewness_10d'] = return_1d.rolling(10).skew()
            group['kurtosis_10d'] = return_1d.rolling(10).kurt()
            
            processed_groups.append(group)
            
        if not processed_groups:
            return df

        combined_df = pd.concat(processed_groups, ignore_index=True)
        
        combined_df = combined_df.replace([np.inf, -np.inf], np.nan)
        combined_df = combined_df.ffill().bfill()
        
        return combined_df

    def add_target_variable(self, df: pd.DataFrame, prediction_days: int = 5) -> pd.DataFrame:
        """
        添加目标变量（标签）
        
        构建机器学习模型的预测目标：未来N天的涨跌情况。
        
        参数：
            df (pd.DataFrame): 包含收盘价的数据
            prediction_days (int): 预测天数，默认5天
        
        返回：
            pd.DataFrame: 添加目标变量后的数据
        
        添加的列：
            - future_return: 未来N天的收益率
            - target: 目标标签（1=上涨，0=下跌）
        
        注意：
            - 使用 shift(-prediction_days) 获取未来数据
            - 最后 prediction_days 天的数据会因为无法获取未来数据而被删除
            - target=1 表示未来N天收益率为正（上涨）
            - target=0 表示未来N天收益率为负或零（下跌/持平）
        """
        df = df.copy()
        
        if 'stock_code' not in df.columns:
            logger.warning("Column 'stock_code' not found in DataFrame. Target variable will be calculated globally (not recommended for multiple stocks).")
            return df

        grouped = df.groupby('stock_code')
        processed_groups = []
        
        for stock_code, group in tqdm(grouped, desc="Adding target variable"):
            group = group.copy().sort_values('date').reset_index(drop=True)
            
            group['future_return'] = group['close'].shift(-prediction_days) / group['close'] - 1
            group['target'] = (group['future_return'] > 0).astype(int)
            
            processed_groups.append(group)
        
        if not processed_groups:
            return df
            
        combined_df = pd.concat(processed_groups, ignore_index=True)
        combined_df = combined_df.dropna(subset=['target'])
        
        return combined_df

    def prepare_features(self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Optional[pd.Series], List[str]]:
        """
        准备特征矩阵和目标变量
        
        从处理后的数据中提取特征矩阵X和目标变量y，用于模型训练。
        
        参数：
            df (pd.DataFrame): 处理后的数据
            feature_cols (List[str], optional): 指定特征列，默认自动选择
        
        返回：
            Tuple[pd.DataFrame, pd.Series, List[str]]:
                - X: 特征矩阵
                - y: 目标变量（如果存在）
                - feature_cols: 特征列名列表
        
        自动排除的列：
            - stock_code: 股票代码（标识符）
            - date: 日期（时间标识）
            - open, high, low, close: 原始价格（容易导致数据泄露）
            - volume, amount: 原始成交数据
            - future_return: 未来收益率（会导致数据泄露）
            - target: 目标变量本身
        
        注意：
            - 无穷值会被替换为NaN
            - NaN值会被填充为该列的均值，如果均值也是NaN则填充0
        """
        if feature_cols is None:
            exclude_cols = ['stock_code', 'date', '日期', 'open', 'high', 'low', 'close', 'volume', 'amount', 
                           'amplitude', 'pct_change', 'change', 'turnover', 'future_return', 'target']
            feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        y = df['target'].copy() if 'target' in df.columns else None
        
        X = X.replace([np.inf, -np.inf], np.nan)
        
        numeric_cols = X.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
        if not numeric_cols.empty:
            X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())
            
        X = X.fillna(0)
        
        return X, y, feature_cols


class DataPreprocessor:
    """
    数据预处理类
    
    负责数据的标准化、分割和异常值处理。
    
    主要功能：
    1. 特征标准化（Z-score标准化）
    2. 数据集分割（训练集/测试集）
    3. 异常值处理（基于标准差的截断）
    
    使用示例：
        >>> preprocessor = DataPreprocessor()
        >>> X_train_scaled, X_test_scaled = preprocessor.normalize_features(X_train, X_test)
        >>> train_df, test_df = preprocessor.split_data(df, split_ratio=0.8)
    """
    
    def __init__(self):
        """初始化数据预处理器"""
        self.scaler = None

    def normalize_features(self, X_train: np.ndarray, X_test: Optional[np.ndarray] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        特征标准化
        
        使用StandardScaler进行Z-score标准化，使特征均值为0，标准差为1。
        
        参数：
            X_train (np.ndarray): 训练集特征
            X_test (np.ndarray, optional): 测试集特征
        
        返回：
            如果只提供X_train：返回标准化后的训练集
            如果同时提供X_test：返回 (标准化训练集, 标准化测试集)
        
        注意：
            - scaler在训练集上fit，然后transform训练集和测试集
            - 这样可以避免数据泄露（测试集信息泄露到训练过程）
        
        标准化公式：
            X_scaled = (X - mean) / std
        """
        from sklearn.preprocessing import StandardScaler
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        else:
            return X_train_scaled

    def split_data(self, df: pd.DataFrame, split_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        按时间分割数据集
        
        将数据按时间顺序分割为训练集和测试集。
        
        参数：
            df (pd.DataFrame): 完整数据集
            split_ratio (float): 训练集比例，默认0.8
        
        返回：
            Tuple[pd.DataFrame, pd.DataFrame]: (训练集, 测试集)
        
        注意：
            - 时间序列数据不能随机分割，必须按时间顺序分割
            - 这样可以模拟真实的预测场景（用过去预测未来）
        """
        df = df.sort_values('date')
        split_idx = int(len(df) * split_ratio)
        
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        return train_df, test_df

    def remove_outliers(self, df: pd.DataFrame, columns: List[str], n_std: int = 3) -> pd.DataFrame:
        """
        移除异常值
        
        使用标准差方法识别并截断异常值。
        
        参数：
            df (pd.DataFrame): 数据
            columns (List[str]): 需要处理的列
            n_std (int): 标准差倍数，默认3（即3σ原则）
        
        返回：
            pd.DataFrame: 处理后的数据
        
        注意：
            - 使用clip方法截断，而不是删除行
            - 3σ原则：约99.7%的数据落在均值±3标准差范围内
        """
        df = df.copy()
        
        for col in columns:
            mean = df[col].mean()
            std = df[col].std()
            lower_bound = mean - n_std * std
            upper_bound = mean + n_std * std
            
            df[col] = df[col].clip(lower_bound, upper_bound)
        
        return df
