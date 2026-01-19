import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import joblib
from pathlib import Path
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from data_fetcher import StockDataFetcher, FeatureEngineer, DataPreprocessor
from ml_models import StockSelectionModel, EnsembleModel, HyperparameterTuner
from backtest import BacktestEngine, PerformanceEvaluator, BenchmarkComparator
import config
from help_module import show_help_page
from feedback_module import show_feedback_page
from storage import storage
import tracking
import admin_dashboard

CONFIG_FILE = Path("backtest_config.json")

def load_backtest_config():
    default_config = {
        'probability_threshold': 0.55,
        'sell_threshold': 0.45,
        'stop_loss_threshold': 0.12,
        'max_hold_days': 8,
        'max_position_pct': 20,
        'max_positions': 6
    }
    return storage.load_json(CONFIG_FILE, "backtest_config", default_config)

def save_backtest_config(config):
    storage.save_json(CONFIG_FILE, "backtest_config", config)

st.set_page_config(
    page_title="基于机器学习的量化投资选股系统",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Google Analytics & SEO Injection
ga_id = os.environ.get("GA_TRACKING_ID")
if ga_id:
    st.markdown(
        f"""
        <!-- Google tag (gtag.js) -->
        <script async src="https://www.googletagmanager.com/gtag/js?id={ga_id}"></script>
        <script>
            window.dataLayer = window.dataLayer || [];
            function gtag(){{dataLayer.push(arguments);}}
            gtag('js', new Date());
            gtag('config', '{ga_id}');
        </script>
        """,
        unsafe_allow_html=True
    )

st.title("📈 基于机器学习的量化投资选股系统")
logger.info("Application started successfully.")

st.markdown("""
本系统使用机器学习算法构建量化投资选股模型，提供股票数据获取与处理、选股策略回测与评估等功能。
""")

with st.sidebar:
    st.header("⚙️ 系统设置")
    
    page = st.radio(
        "选择功能模块",
        ["数据管理", "模型训练", "策略回测", "选股预测", "性能分析", "帮助中心", "用户反馈", "网站访问统计"],
        label_visibility="collapsed"
    )
    
    st.divider()
    
    # Render privacy settings
    tracking.render_privacy_settings()
    
    st.divider()
    
    st.subheader("📊 数据参数")
    start_date = st.date_input(
        "开始日期",
        value=datetime.strptime(config.DEFAULT_START_DATE, '%Y-%m-%d').date()
    )
    end_date = st.date_input(
        "结束日期",
        value=datetime.strptime(config.DEFAULT_END_DATE, '%Y-%m-%d').date()
    )
    
    # Auto-load data if exists
    if 'raw_data' not in st.session_state:
        data_path = Path("data/stock_data.csv")
        if data_path.exists():
            try:
                fetcher = StockDataFetcher()
                st.session_state['raw_data'] = fetcher.load_data('stock_data.csv')
                # logger.info("Auto-loaded stock_data.csv")
            except Exception as e:
                logger.error(f"Failed to auto-load data: {e}")

    if 'processed_data' not in st.session_state:
        processed_path = Path("data/processed_data.csv")
        if processed_path.exists():
            try:
                fetcher = StockDataFetcher()
                st.session_state['processed_data'] = fetcher.load_data('processed_data.csv')
                # logger.info("Auto-loaded processed_data.csv")
            except Exception as e:
                logger.error(f"Failed to auto-load processed data: {e}")

    st.subheader("🎯 模型参数")
    model_type = st.selectbox(
        "选择模型类型",
        ["random_forest", "xgboost", "lightgbm", "logistic", "svm"],
        help="不同的机器学习算法。RF/XGB/LGBM适合处理非线性关系，Logistic/SVM适合处理线性关系。"
    )
    
    prediction_days = st.slider(
        "预测天数",
        min_value=1,
        max_value=20,
        value=config.PREDICTION_DAYS,
        help="模型预测未来第几天的涨跌情况"
    )
    
    st.subheader("💰 回测参数")
    initial_cash = st.number_input(
        "初始资金",
        min_value=10000,
        max_value=10000000,
        value=config.BACKTEST_PARAMS['initial_cash'],
        help="回测账户的起始资金"
    )
    
    commission = st.number_input(
        "手续费率",
        min_value=0.0,
        max_value=0.1,
        value=config.BACKTEST_PARAMS['commission'],
        format="%.4f",
        help="每笔交易的佣金费率（例如0.0003表示万分之三）"
    )

# Track page visit
tracking.track_page_visit(page)

if page == "数据管理":
    st.header("📊 数据管理")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("股票列表")
        
        # Load user settings if available
        default_stock_list = "\n".join(config.STOCK_LIST)
        if 'restored_stock_list' in st.session_state:
            default_stock_list = st.session_state['restored_stock_list']
        else:
            user_settings = config.load_user_settings()
            if 'stock_list' in user_settings:
                default_stock_list = "\n".join(user_settings['stock_list'])
                
        stock_list_input = st.text_area(
            "输入股票代码（每行一个）",
            value=default_stock_list,
            height=200,
            help="输入要分析的A股代码，如 600519。支持每行输入一个代码。"
        )
        stock_codes = [code.strip() for code in stock_list_input.split("\n") if code.strip()]
        st.info(f"当前共 {len(stock_codes)} 只股票")
    
    with col2:
        st.subheader("数据操作")
        
        # Load user settings
        user_settings = config.load_user_settings()
        
        if st.button("📥 获取股票数据", use_container_width=True):
            with st.spinner("正在获取股票数据..."):
                fetcher = StockDataFetcher()
                df = fetcher.fetch_multiple_stocks(
                    stock_codes,
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d')
                )
                
                if not df.empty:
                    fetcher.save_data(df, 'stock_data.csv')
                    st.success(f"成功获取 {len(df)} 条数据")
                    
                    st.session_state['raw_data'] = df
                    
                    # Save user settings (stock list)
                    user_settings['stock_list'] = stock_codes
                    config.save_user_settings(user_settings)
                else:
                    st.error("获取数据失败")
        
        if st.button("🔧 处理数据特征", use_container_width=True):
            if 'raw_data' in st.session_state:
                with st.spinner("正在处理数据特征..."):
                    engineer = FeatureEngineer()
                    df = st.session_state['raw_data'].copy()
                    
                    df = engineer.add_technical_indicators(df)
                    df = engineer.add_return_features(df)
                    df = engineer.add_target_variable(df, prediction_days)
                    
                    fetcher = StockDataFetcher()
                    fetcher.save_data(df, 'processed_data.csv')
                    
                    st.session_state['processed_data'] = df
                    st.success("数据处理完成")
            else:
                st.warning("请先获取股票数据")
        
        st.divider()
        st.subheader("💾 数据备份与恢复")
        col_backup1, col_backup2 = st.columns(2)
        with col_backup1:
            if st.button("备份设置", use_container_width=True):
                user_settings['stock_list'] = stock_codes
                config.save_user_settings(user_settings)
                st.success("设置已备份")
        with col_backup2:
            if st.button("恢复设置", use_container_width=True):
                loaded_settings = config.load_user_settings()
                if 'stock_list' in loaded_settings:
                    st.session_state['restored_stock_list'] = "\n".join(loaded_settings['stock_list'])
                    st.success("设置已恢复，请刷新页面")
                else:
                    st.warning("未找到备份设置")
    
    st.divider()
    
    if 'raw_data' in st.session_state:
        st.subheader("原始数据预览")
        st.dataframe(st.session_state['raw_data'].head(100), use_container_width=True)
        
        st.subheader("数据统计")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("总数据量", f"{len(st.session_state['raw_data']):,}")
        col2.metric("股票数量", f"{st.session_state['raw_data']['stock_code'].nunique()}")
        col3.metric("日期范围", f"{st.session_state['raw_data']['date'].min().date()} 至 {st.session_state['raw_data']['date'].max().date()}")
        col4.metric("数据列数", len(st.session_state['raw_data'].columns))
    
    if 'processed_data' in st.session_state:
        st.subheader("处理后数据预览")
        st.dataframe(st.session_state['processed_data'].head(100), use_container_width=True)
        
        st.subheader("特征统计")
        feature_cols = [col for col in st.session_state['processed_data'].columns 
                       if col not in ['stock_code', 'date', 'target', 'future_return']]
        st.info(f"共 {len(feature_cols)} 个特征")

elif page == "模型训练":
    st.header("🎯 模型训练")
    
    if 'processed_data' not in st.session_state:
        st.warning("请先在数据管理页面获取并处理数据")
    else:
        df = st.session_state['processed_data']
        
        # Check if target column exists
        if 'target' not in df.columns:
            st.error("⚠️ 检测到数据中缺失目标变量 (target)。")
            st.info("请前往【数据管理】页面，点击【🔧 处理数据特征】按钮，以生成训练所需的目标变量。")
            st.stop()
            
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("训练参数")
            test_size = st.slider("测试集比例", 0.1, 0.4, 0.2, help="将多少比例的数据留作测试验证，不参与训练")
            use_cross_validation = st.checkbox("使用交叉验证", help="使用K折交叉验证来评估模型稳定性")
            cv_folds = st.slider("交叉验证折数", 3, 10, 5) if use_cross_validation else 5
        
        with col2:
            st.subheader("超参数调优")
            use_grid_search = st.checkbox("使用网格搜索", help="自动搜索最佳的模型参数组合，会增加训练时间")
            if use_grid_search:
                st.info("将自动搜索最优超参数")
        
        if st.button("🚀 开始训练模型", use_container_width=True):
            with st.spinner("正在训练模型..."):
                preprocessor = DataPreprocessor()
                engineer = FeatureEngineer()
                
                X, y, feature_cols = engineer.prepare_features(df)
                
                split_idx = int(len(df) * (1 - test_size))
                train_df = df.iloc[:split_idx]
                test_df = df.iloc[split_idx:]
                
                X_train = X.iloc[:split_idx]
                X_test = X.iloc[split_idx:]
                
                # Check if y is None (in case target col was missing or something)
                if y is not None:
                    y_train = y.iloc[:split_idx]
                    y_test = y.iloc[split_idx:]
                else:
                    st.error("无法获取目标变量(target)，请检查数据处理步骤。")
                    st.stop()
                
                X_train_scaled, X_test_scaled = preprocessor.normalize_features(X_train, X_test)
                
                model = StockSelectionModel(model_type=model_type)
                
                if use_grid_search:
                    tuner = HyperparameterTuner(model_type=model_type)
                    model.model = tuner.tune(X_train_scaled, y_train, cv=cv_folds)
                else:
                    model.train(X_train_scaled, y_train)
                
                model.save_model()
                
                st.session_state['model'] = model
                st.session_state['preprocessor'] = preprocessor
                st.session_state['feature_cols'] = feature_cols
                st.session_state['X_test'] = X_test_scaled
                st.session_state['y_test'] = y_test
                
                st.success("模型训练完成！")
        
        if 'model' in st.session_state:
            st.divider()
            
            st.subheader("📈 模型评估结果")
            model = st.session_state['model']
            metrics = model.evaluate(st.session_state['X_test'], st.session_state['y_test'])
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("准确率", f"{metrics['accuracy']:.4f}")
            col2.metric("精确率", f"{metrics['precision']:.4f}")
            col3.metric("召回率", f"{metrics['recall']:.4f}")
            col4.metric("F1分数", f"{metrics['f1_score']:.4f}")
            
            if metrics['roc_auc']:
                st.metric("AUC", f"{metrics['roc_auc']:.4f}")
            
            st.subheader("混淆矩阵")
            cm = metrics['confusion_matrix']
            fig = px.imshow(
                cm,
                labels=dict(x="预测", y="实际", color="数量"),
                x=['负类', '正类'],
                y=['负类', '正类'],
                text_auto=True,
                aspect="auto",
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("特征重要性")
            if model.feature_importance is not None:
                importance_df = model.get_feature_importance(st.session_state['feature_cols'])
                importance_df = importance_df.head(20)
                
                fig = px.bar(
                    importance_df,
                    x='importance',
                    y='feature',
                    orientation='h',
                    title='Top 20 特征重要性'
                )
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)

elif page == "策略回测":
    st.header("⏱️ 策略回测")
    
    if 'model' not in st.session_state:
        st.warning("请先训练模型")
    else:
        model = st.session_state['model']
        preprocessor = st.session_state['preprocessor']
        feature_cols = st.session_state['feature_cols']
        
        backtest_config = load_backtest_config()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("回测设置")
            backtest_start = st.date_input(
                "回测开始日期",
                value=(datetime.now() - timedelta(days=365)).date(),
                help="回测的起始日期"
            )
            backtest_end = st.date_input(
                "回测结束日期",
                value=datetime.now().date(),
                help="回测的结束日期"
            )
        
        with col2:
            st.subheader("交易策略参数")
            st.markdown("### 买入设置")
            probability_threshold = st.number_input(
                "买入概率阈值",
                min_value=0.01,
                max_value=0.99,
                value=backtest_config['probability_threshold'],
                step=0.01,
                format="%.3f",
                help="模型预测概率高于此值时才买入（范围：0.01-0.99，支持任意小数值）"
            )
            st.caption("💡 建议：保守策略使用0.60-0.70，平衡策略使用0.50-0.60，激进策略使用0.30-0.50")
            
            st.markdown("### 卖出设置")
            sell_threshold = st.number_input(
                "卖出概率阈值",
                min_value=0.01,
                max_value=0.99,
                value=backtest_config['sell_threshold'],
                step=0.01,
                format="%.3f",
                help="模型预测概率低于此值时卖出（范围：0.01-0.99，支持任意小数值）"
            )
            stop_loss_threshold = st.number_input(
                "止损阈值 (%)",
                min_value=0.01,
                max_value=0.50,
                value=backtest_config['stop_loss_threshold'],
                step=0.01,
                format="%.2f",
                help="亏损超过此比例时触发止损（范围：1%-50%）"
            )
            st.caption("💡 建议：保守策略使用5%-10%，平衡策略使用10%-15%，激进策略使用15%-25%")
            
            trailing_stop_pct = st.number_input(
                "移动止盈回撤阈值 (%)",
                min_value=0.01,
                max_value=0.30,
                value=0.08,
                step=0.01,
                format="%.2f",
                help="当价格从持仓期间最高点回落超过此比例时卖出止盈"
            )
            
            st.markdown("### 持仓管理")
            max_hold_days = st.number_input(
                "最大持仓天数",
                min_value=1,
                max_value=30,
                value=backtest_config['max_hold_days'],
                step=1,
                help="持有股票的最长天数（范围：1-30天）"
            )
            st.caption("💡 建议：短线策略使用3-5天，中线策略使用5-10天，长线策略使用10-20天")
            
            max_position_pct = st.number_input(
                "单只股票最大仓位 (%)",
                min_value=5,
                max_value=50,
                value=backtest_config['max_position_pct'],
                step=1,
                help="单只股票占最大资金的比例（范围：5%-50%）"
            )
            st.caption("💡 建议：保守策略使用10%-15%，平衡策略使用15%-25%，激进策略使用25%-40%")
            
            max_positions = st.number_input(
                "最大持仓数量",
                min_value=1,
                max_value=20,
                value=backtest_config['max_positions'],
                step=1,
                help="同时持有的最大股票数量（范围：1-20只）"
            )
            st.caption("💡 建议：保守策略使用3-5只，平衡策略使用5-8只，激进策略使用8-15只")
        
        st.markdown("---")
        
        col_param1, col_param2 = st.columns(2)
        with col_param1:
            if st.button("🔄 开始回测", use_container_width=True, type="primary"):
                pass
        with col_param2:
            if st.button("💾 保存参数配置", use_container_width=True):
                config = {
                    'probability_threshold': probability_threshold,
                    'sell_threshold': sell_threshold,
                    'stop_loss_threshold': stop_loss_threshold,
                    'max_hold_days': max_hold_days,
                    'max_position_pct': max_position_pct,
                    'max_positions': max_positions
                }
                save_backtest_config(config)
                st.success("✅ 参数配置已保存！")
        
        if st.button("🔄 开始回测", use_container_width=True, key="run_backtest"):
            with st.spinner("正在进行回测..."):
                df = st.session_state['processed_data'].copy()
                
                backtest_engine = BacktestEngine(
                    initial_cash=initial_cash,
                    commission=commission,
                    buy_threshold=probability_threshold,
                    sell_threshold=sell_threshold,
                    stop_loss_threshold=stop_loss_threshold,
                    max_hold_days=max_hold_days,
                    max_position_pct=max_position_pct / 100,
                    max_positions=max_positions
                )
                backtest_engine.trailing_stop_pct = trailing_stop_pct
                
                results = backtest_engine.run_backtest(
                    df,
                    model,
                    feature_cols,
                    start_date=backtest_start.strftime('%Y-%m-%d'),
                    end_date=backtest_end.strftime('%Y-%m-%d')
                )
                
                # Fetch benchmark
                try:
                    fetcher = StockDataFetcher()
                    index_df = fetcher.fetch_index_data(backtest_start.strftime('%Y-%m-%d'), backtest_end.strftime('%Y-%m-%d'), "000300")
                    if index_df is not None:
                        results['benchmark_data'] = index_df
                except Exception as e:
                    logger.warning(f"Could not fetch benchmark: {e}")
                
                st.session_state['backtest_results'] = results
                
                st.success("回测完成！")
                
                if results['total_trades'] == 0:
                    st.warning("⚠️ 回测期间未产生任何交易")
                    st.info("""
                    **可能的原因：**
                    1. 买入概率阈值过高 - 尝试降低到 0.4 或更低
                    2. 模型预测概率普遍较低 - 考虑重新训练模型
                    3. 回测日期范围数据不足 - 检查数据是否包含有效日期
                    4. 资金不足 - 检查初始资金设置
                    
                    **建议：**
                    - 降低买入概率阈值
                    - 增加回测日期范围
                    - 检查模型训练质量
                    - 运行诊断工具: `python diagnose_backtest.py`
                    """)
                else:
                    st.success(f"✅ 回测完成，共产生 {results['total_trades']} 笔交易")
        
        if 'backtest_results' in st.session_state:
            results = st.session_state['backtest_results']
            
            st.divider()
            
            st.subheader("💰 回测结果概览")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("总收益率", f"{results['total_return']:.2f}%")
            col2.metric("最终资金", f"¥{results['final_value']:,.2f}")
            col3.metric("夏普比率", f"{results['sharpe_ratio']:.4f}")
            col4.metric("最大回撤", f"{results['max_drawdown']:.2f}%")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("交易次数", results['total_trades'])
            col2.metric("胜率", f"{results.get('win_rate', 0):.2f}%")
            col3.metric("平均盈亏", f"¥{results.get('average_profit', 0):.2f}")
            col4.metric("盈亏比", f"{results.get('profit_factor', 0):.2f}")
            
            st.subheader("📈 资金曲线")
            portfolio = results['portfolio']
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=portfolio['date'],
                y=portfolio['value'],
                mode='lines',
                name='策略资金',
                line=dict(color='blue', width=2)
            ))
            
            if 'benchmark_data' in results:
                bench = results['benchmark_data']
                # Fix: Merge using the same datetime format/type
                # Ensure date columns are datetime objects
                portfolio['date'] = pd.to_datetime(portfolio['date'])
                bench['date'] = pd.to_datetime(bench['date'])
                
                # We need to reconstruct benchmark value from returns
                # Start with portfolio initial value
                if 'market_return' in bench.columns:
                     bench = bench.sort_values('date')
                     # Filter bench to match portfolio dates range
                     start_date = portfolio['date'].min()
                     bench = bench[bench['date'] >= start_date]
                     
                     # Calculate cumulative return series
                     initial_value = portfolio['value'].iloc[0]
                     
                     # Reindex to match portfolio dates to handle missing trading days
                     merged = pd.merge(portfolio[['date']], bench[['date', 'market_return']], on='date', how='left')
                     merged['market_return'] = merged['market_return'].fillna(0)
                     
                     # Calculate value curve
                     merged['bench_val'] = initial_value * (1 + merged['market_return']).cumprod()
                     
                     fig.add_trace(go.Scatter(
                        x=merged['date'],
                        y=merged['bench_val'],
                        mode='lines',
                        name='基准(沪深300)',
                        line=dict(color='gray', width=1, dash='dot')
                    ))

            fig.update_layout(
                title='资金曲线',
                xaxis_title='日期',
                yaxis_title='资金',
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("💹 持仓分析")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=portfolio['date'],
                y=portfolio['cash'],
                mode='lines',
                name='现金',
                fill='tozeroy'
            ))
            fig.add_trace(go.Scatter(
                x=portfolio['date'],
                y=portfolio['positions_value'],
                mode='lines',
                name='持仓市值',
                fill='tonexty'
            ))
            fig.update_layout(
                title='现金与持仓分布',
                xaxis_title='日期',
                yaxis_title='金额',
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("📊 交易记录")
            trades = results['trades']
            if not trades.empty:
                st.dataframe(trades, use_container_width=True)
                
                sell_trades = trades[trades['action'] == 'sell']
                if not sell_trades.empty:
                    fig = px.bar(
                        sell_trades,
                        x='date',
                        y='profit',
                        color=sell_trades['profit'] > 0,
                        color_discrete_map={True: 'green', False: 'red'},
                        title='交易盈亏分布'
                    )
                    st.plotly_chart(fig, use_container_width=True)

elif page == "选股预测":
    st.header("🔮 选股预测")
    
    if 'model' not in st.session_state:
        st.warning("请先训练模型")
    else:
        model = st.session_state['model']
        preprocessor = st.session_state['preprocessor']
        feature_cols = st.session_state['feature_cols']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("预测设置")
            prediction_date = st.date_input(
                "预测日期",
                value=datetime.now().date(),
                help="基于该日期的数据进行预测"
            )
            top_n = st.slider(
                "推荐股票数量",
                min_value=5,
                max_value=50,
                value=10,
                help="最终展示的预测上涨概率最高的股票数量"
            )
        
        with col2:
            st.subheader("筛选条件")
            min_probability = st.slider(
                "最小预测概率",
                min_value=0.5,
                max_value=0.9,
                value=0.6,
                step=0.05,
                help="只推荐预测上涨概率高于此值的股票"
            )
        
        if st.button("🔍 开始预测", use_container_width=True):
            try:
                with st.spinner("正在进行预测..."):
                    logger.info("=" * 60)
                    logger.info("开始选股预测")
                    logger.info("=" * 60)
                    
                    df = st.session_state['processed_data'].copy()
                    logger.info(f"数据形状: {df.shape}")
                    
                    if df.empty:
                        st.error("❌ 数据为空，无法进行预测！")
                        logger.error("数据为空")
                    
                    model = st.session_state['model']
                    preprocessor = st.session_state['preprocessor']
                    feature_cols = st.session_state['feature_cols']
                    
                    logger.info(f"模型类型: {model.model_type}")
                    logger.info(f"特征数量: {len(feature_cols)}")
                    
                    latest_data = df.groupby('stock_code').last().reset_index()
                    logger.info(f"最新数据行数: {len(latest_data)}")
                    
                    if latest_data.empty:
                        st.error("❌ 无法获取最新数据！")
                        logger.error("最新数据为空")
                    
                    X = latest_data[feature_cols].values
                    logger.info(f"特征数据形状: {X.shape}")
                    
                    if preprocessor.scaler is None:
                        logger.warning("数据预处理器未训练，将使用原始特征")
                        X_scaled = X
                    else:
                        logger.info("使用训练好的数据预处理器")
                        X_scaled = preprocessor.scaler.transform(X)
                    
                    logger.info(f"缩放后特征形状: {X_scaled.shape}")
                    
                    predictions = model.predict(X_scaled)
                    probabilities = model.predict_proba(X_scaled)[:, 1]
                    
                    logger.info(f"预测完成: 上涨={sum(predictions)}, 下跌={len(predictions)-sum(predictions)}")
                    logger.info(f"概率统计: 均值={probabilities.mean():.4f}, 最大={probabilities.max():.4f}, 最小={probabilities.min():.4f}")
                    
                    latest_data['prediction'] = predictions
                    latest_data['probability'] = probabilities
                    
                    logger.info(f"筛选条件: 预测=1 且 概率>={min_probability}")
                    
                    recommended_stocks = latest_data[
                        (latest_data['prediction'] == 1) & 
                        (latest_data['probability'] >= min_probability)
                    ].sort_values('probability', ascending=False).head(top_n)
                    
                    logger.info(f"推荐股票数量: {len(recommended_stocks)}")
                    
                    if len(recommended_stocks) > 0:
                        logger.info("推荐股票详情:")
                        for idx, stock in recommended_stocks.iterrows():
                            logger.info(f"  {idx+1}. {stock['stock_code']}: 概率={stock['probability']:.4f}")
                    else:
                        logger.warning("未找到符合条件的推荐股票")
                    
                    if len(recommended_stocks) > 0:
                        st.session_state['predicted_stocks'] = recommended_stocks
                        logger.info(f"已保存到session_state: predicted_stocks形状={recommended_stocks.shape}")
                        
                        # Save prediction to storage for history
                        try:
                            # Ensure columns exist before saving
                            cols_to_save = ['stock_code', 'probability', 'close']
                            cols_present = [c for c in cols_to_save if c in recommended_stocks.columns]
                            save_data = recommended_stocks[cols_present].to_dict(orient='records')
                            
                            current_date_str = datetime.now().strftime('%Y-%m-%d')
                            storage.save_predictions(current_date_str, save_data)
                            logger.info(f"Saved predictions for {current_date_str}")
                        except Exception as e:
                            logger.error(f"Failed to save prediction history: {e}")
                        
                        st.success(f"✅ 预测完成！找到 {len(recommended_stocks)} 只推荐股票")
                        st.info(f"💡 预测概率范围: {probabilities.min():.2f} - {probabilities.max():.2f}")
                    else:
                        st.warning("⚠️ 未找到符合条件的推荐股票")
                        st.info(f"💡 预测概率范围: {probabilities.min():.2f} - {probabilities.max():.2f}")
                        st.info(f"💡 建议: 尝试降低最小概率阈值（当前: {min_probability}）")
                        st.info("""
                        **可能的原因：**
                        1. 预测概率普遍较低 - 尝试降低最小概率阈值
                        2. 特征数据存在问题 - 检查数据质量
                        3. 模型预测准确率较低 - 考虑重新训练模型
                        4. 筛选条件过于严格 - 调整最小概率阈值或增加推荐数量
                        5. 数据预处理问题 - 确保数据预处理器已正确训练
                        6. 特征列不匹配 - 检查特征列是否正确
                        """)
                        
            except Exception as e:
                logger.error(f"预测过程中发生错误: {e}", exc_info=True)
                st.error(f"❌ 预测失败: {str(e)}")
                st.info("请检查以下内容：")
                st.info("1. 模型是否已正确训练")
                st.info("2. 数据预处理是否已完成")
                st.info("3. 特征列是否正确匹配")
                st.info("4. 查看日志了解详细错误信息")
                st.info("5. 运行诊断工具: python diagnose_prediction.py")
        
    if 'predicted_stocks' in st.session_state:
        recommended_stocks = st.session_state['predicted_stocks']
        
        st.divider()
        
        st.subheader("📋 推荐股票列表")
        
        if recommended_stocks.empty:
            st.info("没有找到符合条件的推荐股票")
        else:
            # Ensure bollinger_width exists
            if 'bollinger_width' not in recommended_stocks.columns:
                 if 'bb_width' in recommended_stocks.columns:
                     recommended_stocks['bollinger_width'] = recommended_stocks['bb_width']
                 else:
                     recommended_stocks['bollinger_width'] = 0.0

            # User customization
            with st.expander("🛠️ 自定义显示列"):
                # Add fundamental cols to options
                all_cols = ['stock_code', 'close', 'probability', 'prediction', 
                           'pe_ratio', 'pb_ratio', 'total_market_cap',
                           'ma5', 'ma20', 'rsi', 'volume_ratio', 'macd', 'bollinger_width']
                
                # Filter cols that actually exist in dataframe to prevent KeyError
                valid_cols = [c for c in all_cols if c in recommended_stocks.columns]
                
                default_cols = ['stock_code', 'close', 'probability', 'pe_ratio', 'rsi', 'volume_ratio']
                # Ensure defaults are valid
                default_cols = [c for c in default_cols if c in valid_cols]
                
                selected_cols = st.multiselect("选择要显示的列", valid_cols, default=default_cols)
            
            # --- 分层展示：第一层 简洁结论 ---
            st.markdown("### 1. 核心预测结论")
            top_stock = recommended_stocks.iloc[0]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("首选推荐", top_stock['stock_code'], delta="强烈推荐" if top_stock['probability'] > 0.8 else "推荐")
            with col2:
                st.metric("预测上涨概率 (置信度)", f"{top_stock['probability']:.2%}", help="模型预测未来上涨的可能性，越高越好")
            with col3:
                st.metric("当前价格", f"{top_stock['close']:.2f}")

            # --- 分层展示：第二层 关键因素摘要 ---
            st.markdown("### 2. 关键驱动因素")
            st.info("以下特征是模型判断该股票上涨的主要依据：")
            
            kpi_cols = st.columns(4)
            kpi_cols[0].metric("RSI (相对强弱)", f"{top_stock['rsi']:.2f}", help=">70 超买, <30 超卖. 50左右为中性.")
            kpi_cols[1].metric("量比", f"{top_stock['volume_ratio']:.2f}", help=">1 表示放量, <1 表示缩量.")
            
            # Show PE if available
            if 'pe_ratio' in top_stock and pd.notna(top_stock['pe_ratio']):
                 kpi_cols[2].metric("市盈率 (PE)", f"{top_stock['pe_ratio']:.1f}", help="越低通常越有价值，但需结合行业看")
            else:
                 kpi_cols[2].metric("MA5", f"{top_stock['ma5']:.2f}")
                 
            if 'market_return' in top_stock and pd.notna(top_stock['market_return']):
                kpi_cols[3].metric("大盘趋势", f"{top_stock['market_return']:.2%}", delta_color="normal")
            else:
                kpi_cols[3].metric("MACD", f"{top_stock['macd']:.2f}")

            # --- 分层展示：第三层 详细分析 & 可视化 ---
            st.markdown("### 3. 深度分析与可视化")
            
            tab_list, tab_chart, tab_history = st.tabs(["📜 详细数据列表", "📈 趋势可视化", "🔙 历史回看验证"])
            
            with tab_list:
                display_df = recommended_stocks[selected_cols].copy()
                # Rename for readability
                col_map = {
                    'stock_code': '股票代码', 'close': '收盘价', 'probability': '预测概率',
                    'prediction': '预测方向', 'ma5': '5日均线', 'ma20': '20日均线',
                    'rsi': 'RSI指标', 'volume_ratio': '量比', 'macd': 'MACD', 'bollinger_width': '布林带宽度',
                    'pe_ratio': '市盈率(PE)', 'pb_ratio': '市净率(PB)', 'total_market_cap': '总市值'
                }
                display_df = display_df.rename(columns=col_map)
                st.dataframe(display_df, use_container_width=True)
            
            with tab_chart:
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.bar(
                        recommended_stocks.head(10),
                        x='stock_code',
                        y='probability',
                        title='Top 10 推荐股票预测概率',
                        labels={'probability': '预测概率', 'stock_code': '股票代码'},
                        color='probability',
                        color_continuous_scale='Viridis'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.scatter(
                        recommended_stocks,
                        x='rsi',
                        y='probability',
                        size='volume_ratio',
                        color='stock_code',
                        title='RSI vs 预测概率 (气泡大小=量比)',
                        labels={'rsi': 'RSI', 'probability': '预测概率'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
            with tab_history:
                st.markdown("#### 历史验证")
                st.caption("查看模型在历史日期的预测表现。")
                
                # Date selection for historical verification
                verify_date = st.date_input("选择历史日期进行回测", 
                                          value=datetime.now().date() - timedelta(days=5),
                                          max_value=datetime.now().date() - timedelta(days=1))
                
                if st.button("🔍 验证该日预测准确率"):
                    date_str = verify_date.strftime('%Y-%m-%d')
                    
                    # 1. Try to load saved predictions
                    saved_preds = storage.load_predictions(date_str)
                    
                    if saved_preds:
                        st.success(f"找到 {date_str} 的历史预测记录")
                        # Calculate accuracy based on current prices (realized return)
                        # This requires fetching current data for the stocks in saved_preds
                        # For now, just show the saved prediction count
                        st.json(saved_preds)
                    else:
                        st.warning(f"未找到 {date_str} 的历史预测记录。")
                        st.info("提示：系统会自动保存每天的预测结果。您也可以现在手动运行回测来模拟当时的预测。")
                        
                        # Optional: Run simulation for that date (requires advanced logic)

            # --- 用户操作指引 & 解释说明 ---
            with st.expander("📖 预测结果解读指南 (点击展开)"):
                st.markdown("""
                ### 如何看懂预测结果？
                
                1.  **预测概率 (Probability)**: 模型认为该股票在未来（如5天后）上涨的可能性。
                    *   **> 0.8**: 极高置信度，强烈看涨。
                    *   **0.6 - 0.8**: 较高置信度，看涨。
                    *   **< 0.5**: 看跌或震荡。
                
                2.  **关键指标解释**:
                    *   **RSI**: 衡量资金进出的强弱。数值在30-70之间属于正常波动，超过70可能回调，低于30可能反弹。
                    *   **量比**: 当日成交量与过去5天平均成交量的比值。量比大说明资金活跃。
                
                3.  **使用建议**:
                    *   不要仅依赖单一指标。结合K线形态和市场热点综合判断。
                    *   建议分散投资，不要重仓单一股票。
                """)

elif page == "性能分析":
    st.header("📊 性能分析")
    
    if 'backtest_results' not in st.session_state:
        st.warning("请先进行策略回测")
    else:
        results = st.session_state['backtest_results']
        portfolio = results['portfolio']
        trades = results['trades']
        
        st.subheader("📈 收益分析")
        
        portfolio['daily_return'] = portfolio['value'].pct_change()
        portfolio['cumulative_return'] = (portfolio['value'] / portfolio['value'].iloc[0] - 1) * 100
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=portfolio['date'],
            y=portfolio['cumulative_return'],
            mode='lines',
            name='累计收益率',
            line=dict(color='blue', width=2)
        ))
        fig.update_layout(
            title='累计收益率曲线',
            xaxis_title='日期',
            yaxis_title='收益率 (%)',
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("📊 收益分布")
        
        daily_returns = portfolio['daily_return'].dropna()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                daily_returns,
                nbins=50,
                title='日收益率分布',
                labels={'value': '日收益率'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(
                y=daily_returns,
                title='日收益率箱线图',
                labels={'y': '日收益率'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("💹 风险指标")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("年化波动率", f"{daily_returns.std() * np.sqrt(252) * 100:.2f}%")
        col2.metric("最大回撤", f"{results['max_drawdown']:.2f}%")
        col3.metric("夏普比率", f"{results['sharpe_ratio']:.4f}")
        col4.metric("索提诺比率", f"{daily_returns.mean() / daily_returns[daily_returns < 0].std() * np.sqrt(252):.4f}")
        
        if not trades.empty:
            st.subheader("📋 交易统计")
            
            sell_trades = trades[trades['action'] == 'sell']
            
            if not sell_trades.empty:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("盈利交易", results['winning_trades'])
                col2.metric("亏损交易", results['losing_trades'])
                col3.metric("平均盈利", f"¥{sell_trades[sell_trades['profit'] > 0]['profit'].mean():.2f}")
                col4.metric("平均亏损", f"¥{sell_trades[sell_trades['profit'] <= 0]['profit'].mean():.2f}")
                
                fig = px.histogram(
                    sell_trades,
                    x='profit',
                    nbins=30,
                    title='交易盈亏分布',
                    color=sell_trades['profit'] > 0,
                    color_discrete_map={True: 'green', False: 'red'}
                )
                st.plotly_chart(fig, use_container_width=True)

elif page == "帮助中心":
    show_help_page()

elif page == "用户反馈":
    show_feedback_page()

elif page == "网站访问统计":
    st.header("📊 网站访问统计 (管理员)")
    
    # Simple password protection
    if 'admin_logged_in' not in st.session_state:
        st.session_state['admin_logged_in'] = False
        
    if not st.session_state['admin_logged_in']:
        pwd = st.text_input("请输入管理员密码", type="password")
        if st.button("登录"):
            if pwd == "admin":  # Default password, user should change this
                st.session_state['admin_logged_in'] = True
                st.rerun()
            else:
                st.error("密码错误")
    
    if st.session_state['admin_logged_in']:
        if st.button("退出登录"):
            st.session_state['admin_logged_in'] = False
            st.rerun()
        admin_dashboard.show_dashboard()

st.divider()
st.markdown("""
---
**提示**: 本系统仅供学习和研究使用，不构成投资建议。投资有风险，入市需谨慎。
""")
