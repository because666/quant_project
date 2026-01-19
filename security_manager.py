
import sqlite3
import re
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import html
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class SecurityManager:
    def __init__(self, db_path: str = "data/security.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()
        
        # In-memory rate limiting: {ip: [timestamps]}
        self.request_history = {}
        self.blacklist = set()
        
        # WAF Rules (Regex)
        self.rules = {
            'sql_injection': r"(?i)(\b(select|union|insert|update|delete|drop|alter)\b.*\b(from|into|table|database)\b)|(--)|(\b(or|and)\b\s+[\w-]+\s*(=|like|>|<)\s*[\w-])",
            'xss': r"(?i)(<script>|javascript:|onload=|onerror=|onclick=|<iframe|<object)",
            'path_traversal': r"(\.\./|\.\.\\)",
            'command_injection': r"(;|\||`|\$\()",
        }
        
        # Bot signatures
        self.bot_signatures = [
            'headless', 'selenium', 'phantomjs', 'puppeteer', 'webdriver', 'bot', 'crawler', 'spider'
        ]

    def get_connection(self):
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def init_db(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS attack_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            ip_address TEXT,
            attack_type TEXT,
            payload TEXT,
            action TEXT,
            risk_level TEXT,
            user_agent TEXT
        )
        ''')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_attack_time ON attack_logs(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_attack_type ON attack_logs(attack_type)')
        
        conn.commit()
        conn.close()

    def log_attack(self, ip: str, attack_type: str, payload: str, action: str, risk_level: str, user_agent: str = ""):
        """Log an attack event."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute('''
            INSERT INTO attack_logs (timestamp, ip_address, attack_type, payload, action, risk_level, user_agent)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (datetime.now(), ip, attack_type, payload, action, risk_level, user_agent))
            conn.commit()
            conn.close()
            logger.warning(f"Security Alert: {attack_type} from {ip} - {action}")
        except Exception as e:
            logger.error(f"Failed to log attack: {e}")

    def check_rate_limit(self, ip: str, limit: int = 100, window: int = 60) -> bool:
        """
        Check if IP exceeds rate limit.
        limit: max requests
        window: seconds
        """
        if ip in self.blacklist:
            return False
            
        now = time.time()
        if ip not in self.request_history:
            self.request_history[ip] = []
        
        # Clean old requests
        self.request_history[ip] = [t for t in self.request_history[ip] if now - t < window]
        
        # Check limit
        if len(self.request_history[ip]) >= limit:
            self.log_attack(ip, "rate_limit_exceeded", f"Requests: {len(self.request_history[ip])}/{window}s", "block", "medium")
            return False
            
        self.request_history[ip].append(now)
        return True

    def detect_sql_injection(self, text: str) -> bool:
        if not text: return False
        return bool(re.search(self.rules['sql_injection'], text))

    def detect_xss(self, text: str) -> bool:
        if not text: return False
        return bool(re.search(self.rules['xss'], text))

    def detect_bot(self, user_agent: str) -> bool:
        if not user_agent: return False
        ua = user_agent.lower()
        for sig in self.bot_signatures:
            if sig in ua:
                return True
        return False

    def validate_input(self, text: str, ip: str = "unknown", user_agent: str = "") -> Tuple[bool, str]:
        """
        Validate user input against WAF rules.
        Returns (is_safe, message)
        """
        if self.detect_sql_injection(text):
            self.log_attack(ip, "sql_injection", text, "block", "high", user_agent)
            return False, "检测到潜在的 SQL 注入尝试"
            
        if self.detect_xss(text):
            self.log_attack(ip, "xss_attack", text, "block", "high", user_agent)
            return False, "检测到潜在的 XSS 攻击脚本"
            
        return True, "ok"

    def get_attack_stats(self, days: int = 30) -> Dict:
        """Get statistics for dashboard."""
        start_date = datetime.now() - timedelta(days=days)
        conn = self.get_connection()
        
        stats = {}
        
        # Total attacks
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM attack_logs WHERE timestamp >= ?", (start_date,))
        stats['total_attacks'] = cursor.fetchone()[0]
        
        # Attacks by type
        df_type = pd.read_sql_query("SELECT attack_type, COUNT(*) as count FROM attack_logs WHERE timestamp >= ? GROUP BY attack_type", conn, params=(start_date,))
        stats['by_type'] = df_type
        
        # Recent logs
        df_logs = pd.read_sql_query("SELECT * FROM attack_logs WHERE timestamp >= ? ORDER BY timestamp DESC LIMIT 100", conn, params=(start_date,))
        stats['recent_logs'] = df_logs
        
        # Trend
        df_trend = pd.read_sql_query("SELECT date(timestamp) as date, COUNT(*) as count FROM attack_logs WHERE timestamp >= ? GROUP BY date(timestamp)", conn, params=(start_date,))
        stats['trend'] = df_trend
        
        conn.close()
        return stats

# Global instance
security = SecurityManager()
