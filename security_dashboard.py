
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from security_manager import security
from datetime import datetime

def show_security_dashboard():
    st.subheader("🛡️ 安全中心")
    
    # 1. Overview
    stats = security.get_attack_stats(days=30)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("近30天拦截攻击", stats['total_attacks'])
    with col2:
        st.metric("系统防护状态", "运行中 🟢")
    with col3:
        st.metric("防火墙规则库", f"{len(security.rules)} 类")
    with col4:
        st.metric("最后更新", datetime.now().strftime("%H:%M"))

    st.divider()

    col_left, col_right = st.columns(2)

    # 2. Attack Distribution
    with col_left:
        st.subheader("🚨 攻击类型分布")
        if not stats['by_type'].empty:
            fig = px.pie(stats['by_type'], values='count', names='attack_type', hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("暂无攻击记录（系统很安全）")

    # 3. Attack Trend
    with col_right:
        st.subheader("📈 攻击拦截趋势")
        if not stats['trend'].empty:
            stats['trend']['date'] = pd.to_datetime(stats['trend']['date'])
            fig = px.line(stats['trend'], x='date', y='count', markers=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("暂无趋势数据")

    # 4. Detailed Logs
    st.subheader("📝 拦截日志 (最近100条)")
    
    if not stats['recent_logs'].empty:
        logs = stats['recent_logs']
        
        # Filter
        filter_type = st.multiselect("筛选攻击类型", logs['attack_type'].unique())
        if filter_type:
            logs = logs[logs['attack_type'].isin(filter_type)]
            
        st.dataframe(logs, use_container_width=True)
        
        # Export
        st.download_button(
            "📥 导出安全日志",
            logs.to_csv(index=False).encode('utf-8'),
            "security_logs.csv",
            "text/csv"
        )
    else:
        st.info("暂无日志记录")

    # 5. Manual Simulation (Test)
    with st.expander("🧪 安全测试工具 (仅限管理员)"):
        st.warning("此功能用于测试WAF规则是否生效")
        test_input = st.text_input("输入测试Payload (例如尝试输入 <script>alert(1)</script>)")
        if st.button("提交测试"):
            is_safe, msg = security.validate_input(test_input, ip="test_admin", user_agent="manual_test")
            if not is_safe:
                st.error(f"拦截成功！原因: {msg}")
            else:
                st.success("未检测到威胁")
