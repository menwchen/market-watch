import sqlite3
import json
import hashlib
import time
import os
from typing import Any, Optional

from config import Config


class CacheStore:
    """SQLite-based API response cache."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or Config.CACHE_DB_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    ttl_hours REAL NOT NULL
                )
            """)

    @staticmethod
    def _make_key(namespace: str, identifier: str) -> str:
        raw = f"{namespace}:{identifier}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, namespace: str, identifier: str) -> Optional[Any]:
        key = self._make_key(namespace, identifier)
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT value, created_at, ttl_hours FROM cache WHERE key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        value, created_at, ttl_hours = row
        if time.time() - created_at > ttl_hours * 3600:
            self.delete(namespace, identifier)
            return None
        return json.loads(value)

    def set(self, namespace: str, identifier: str, value: Any,
            ttl_hours: float = None):
        key = self._make_key(namespace, identifier)
        ttl = ttl_hours if ttl_hours is not None else Config.CACHE_TTL_HOURS
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, value, created_at, ttl_hours) "
                "VALUES (?, ?, ?, ?)",
                (key, json.dumps(value, default=str), time.time(), ttl),
            )

    def delete(self, namespace: str, identifier: str):
        key = self._make_key(namespace, identifier)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM cache WHERE key = ?", (key,))

    def clear_expired(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM cache WHERE (? - created_at) > (ttl_hours * 3600)",
                (time.time(),),
            )

    def clear_all(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM cache")
