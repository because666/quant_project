
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class AnalyticsManager:
    def __init__(self, db_path: str = "data/analytics.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def get_connection(self):
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def init_db(self):
        """Initialize the analytics database table."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Create visits table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS visits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            visitor_id TEXT,
            ip_address TEXT,
            url_path TEXT,
            user_agent TEXT,
            device_info TEXT,
            referrer TEXT
        )
        ''')
        
        # Create indices for common query fields
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON visits(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_visitor_id ON visits(visitor_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_url_path ON visits(url_path)')
        
        conn.commit()
        conn.close()

    def anonymize_ip(self, ip: str) -> str:
        """Anonymize IP address (mask last octet)."""
        if not ip or ip == 'unknown':
            return 'unknown'
        try:
            parts = ip.split('.')
            if len(parts) == 4:
                return f"{parts[0]}.{parts[1]}.{parts[2]}.0"
            # Handle IPv6 simply by hashing or partial mask
            if ':' in ip:
                return ip.split(':')[0] + ":****"
            return ip
        except:
            return 'unknown'

    def generate_visitor_id(self, ip: str, user_agent: str) -> str:
        """Generate a unique visitor ID based on IP and User Agent."""
        raw = f"{ip}-{user_agent}"
        return hashlib.md5(raw.encode()).hexdigest()

    def log_visit(self, ip: str, url_path: str, user_agent: str = "", referrer: str = ""):
        """Log a page visit."""
        try:
            # Anonymize IP
            anon_ip = self.anonymize_ip(ip)
            visitor_id = self.generate_visitor_id(ip, user_agent)
            
            # Simple device info parsing (mock)
            device_info = "Unknown"
            if "Mobile" in user_agent:
                device_info = "Mobile"
            elif "Windows" in user_agent:
                device_info = "PC (Windows)"
            elif "Mac" in user_agent:
                device_info = "PC (Mac)"
            elif "Linux" in user_agent:
                device_info = "PC (Linux)"
                
            timestamp = datetime.now()

            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT INTO visits (timestamp, visitor_id, ip_address, url_path, user_agent, device_info, referrer)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (timestamp, visitor_id, anon_ip, url_path, user_agent, device_info, referrer))
            
            conn.commit()
            conn.close()
            # logger.info(f"Logged visit: {url_path} from {anon_ip}")
        except Exception as e:
            logger.error(f"Failed to log visit: {e}")

    def get_total_pv(self) -> int:
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM visits")
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def get_total_uv(self) -> int:
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(DISTINCT visitor_id) FROM visits")
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def get_trend_data(self, days: int = 30) -> pd.DataFrame:
        """Get daily PV and UV for the last N days."""
        start_date = datetime.now() - timedelta(days=days)
        conn = self.get_connection()
        
        query = f'''
        SELECT 
            date(timestamp) as date,
            COUNT(*) as pv,
            COUNT(DISTINCT visitor_id) as uv
        FROM visits
        WHERE timestamp >= ?
        GROUP BY date(timestamp)
        ORDER BY date(timestamp)
        '''
        
        df = pd.read_sql_query(query, conn, params=(start_date,))
        conn.close()
        return df

    def get_top_pages(self, limit: int = 10) -> pd.DataFrame:
        conn = self.get_connection()
        query = f'''
        SELECT url_path, COUNT(*) as visits
        FROM visits
        GROUP BY url_path
        ORDER BY visits DESC
        LIMIT {limit}
        '''
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df

    def get_device_stats(self) -> pd.DataFrame:
        conn = self.get_connection()
        query = '''
        SELECT device_info, COUNT(*) as count
        FROM visits
        GROUP BY device_info
        ORDER BY count DESC
        '''
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
        
# Global instance
analytics = AnalyticsManager()
