
import streamlit as st
from analytics_manager import analytics
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def track_page_visit(page_name: str):
    """
    Track a page visit. Should be called at the top of every page.
    """
    # 1. Check Opt-out status
    if 'analytics_opt_out' not in st.session_state:
        st.session_state['analytics_opt_out'] = False
        # Optional: Show a privacy notice toast on first visit
        if 'privacy_notice_shown' not in st.session_state:
            st.toast("🍪 本网站使用Cookies收集匿名访问数据以优化体验。您可以在侧边栏设置中选择退出。", icon="ℹ️")
            st.session_state['privacy_notice_shown'] = True

    if st.session_state['analytics_opt_out']:
        return

    # 2. Prevent duplicate logging for the same script run (rerun)
    # We use a composite key of (page_name, timestamp_minute) or just session state flag per page load
    # Streamlit reruns the script on interaction. We only want to log "Page Load", not every interaction.
    # But detecting "Page Load" vs "Interaction" is hard.
    # A simple heuristic: Only log if page_name changes or it's a fresh session start.
    
    if 'last_visited_page' not in st.session_state:
        st.session_state['last_visited_page'] = None
    
    # We define a "Visit" as entering a page.
    # If we are already on this page and just interacting, maybe we don't log?
    # But usually people want to count interactions? 
    # Let's count every "Script Run" as a hit? No, that's too much (every button click).
    # Let's count 1 hit per unique page load per session? 
    # Or count 1 hit if `last_visited_page` != `page_name`.
    
    should_log = False
    if st.session_state['last_visited_page'] != page_name:
        should_log = True
        st.session_state['last_visited_page'] = page_name
    
    # Also log if it's been a while (e.g. > 30 mins) - simplified to just page change for now
    
    if should_log:
        try:
            # Gather Info
            # IP/UA are hard to get in pure Streamlit without hacks.
            # We will use placeholders or headers if available (e.g. if deployed behind proxy)
            # For local dev, these will be empty.
            
            headers = {}
            # Try to get headers from browser context (requires streamlit>=1.20 and specific config)
            # This is often unreliable in local mode.
            
            ip = "unknown"
            user_agent = "unknown"
            
            # If running in Zeabur (Docker), we might check env vars or headers
            # Here we just log what we have
            
            analytics.log_visit(
                ip=ip, 
                url_path=page_name, 
                user_agent=user_agent
            )
        except Exception as e:
            logger.error(f"Tracking error: {e}")

def render_privacy_settings():
    """Render privacy settings in the sidebar."""
    with st.sidebar.expander("🔒 隐私设置"):
        st.write("我们需要收集少量匿名数据来改进服务。")
        opt_out = st.checkbox("🚫 退出数据追踪 (Opt-out)", value=st.session_state.get('analytics_opt_out', False))
        if opt_out != st.session_state.get('analytics_opt_out', False):
            st.session_state['analytics_opt_out'] = opt_out
            if opt_out:
                st.success("已停止数据追踪。")
            else:
                st.info("已启用数据追踪。")
