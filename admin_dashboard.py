
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from analytics_manager import analytics
from datetime import datetime

def show_dashboard():
    st.subheader("📈 访问概览")
    
    # 1. Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("总访问量 (PV)", analytics.get_total_pv())
    with col2:
        st.metric("独立访客 (UV)", analytics.get_total_uv())
    with col3:
        # Calculate today's PV
        # This would require a more specific query in analytics_manager
        # For now, placeholder or simple calc
        st.metric("今日访问", "N/A", help="需完善按日查询接口")
    with col4:
        st.metric("系统状态", "在线 🟢")

    st.divider()

    # 2. Trends
    st.subheader("📅 访问趋势 (近30天)")
    trend_df = analytics.get_trend_data(days=30)
    if not trend_df.empty:
        # Convert date to datetime for better plotting
        trend_df['date'] = pd.to_datetime(trend_df['date'])
        
        tab1, tab2 = st.tabs(["PV/UV 趋势图", "数据表"])
        with tab1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=trend_df['date'], y=trend_df['pv'], name='PV (访问量)', mode='lines+markers'))
            fig.add_trace(go.Scatter(x=trend_df['date'], y=trend_df['uv'], name='UV (访客数)', mode='lines+markers'))
            fig.update_layout(hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
        with tab2:
            st.dataframe(trend_df, use_container_width=True)
    else:
        st.info("暂无趋势数据")

    col_left, col_right = st.columns(2)

    # 3. Top Pages
    with col_left:
        st.subheader("🔥 热门页面")
        top_pages = analytics.get_top_pages(limit=10)
        if not top_pages.empty:
            fig = px.bar(top_pages, x='visits', y='url_path', orientation='h', title="访问量 Top 10")
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("暂无数据")

    # 4. Device Stats
    with col_right:
        st.subheader("📱 设备分布")
        device_df = analytics.get_device_stats()
        if not device_df.empty:
            fig = px.pie(device_df, values='count', names='device_info', title="用户设备类型")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("暂无数据")

    # 5. Raw Data (Optional, for admin inspection)
    with st.expander("🕵️ 查看最新访问日志 (最近 50 条)"):
        conn = analytics.get_connection()
        logs = pd.read_sql_query("SELECT * FROM visits ORDER BY timestamp DESC LIMIT 50", conn)
        
        # Export button
        full_logs = pd.read_sql_query("SELECT * FROM visits", conn)
        conn.close()
        
        st.download_button(
            label="📥 导出完整访问日志 (CSV)",
            data=full_logs.to_csv(index=False).encode('utf-8'),
            file_name=f'access_logs_{datetime.now().strftime("%Y%m%d")}.csv',
            mime='text/csv',
        )
        
        st.dataframe(logs)
