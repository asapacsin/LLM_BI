"""SQLite-backed session and message store."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import STATE_DB_PATH


class StateStore:
    """Persist agent sessions, messages, and run metadata locally."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or STATE_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                );
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    query TEXT,
                    intent TEXT,
                    latency_ms REAL,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                );
                """
            )

    def create_session(self, title: str = "New conversation") -> str:
        sid = str(uuid.uuid4())[:8]
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (sid, title, now, now),
            )
        self._export_json(sid)
        return sid

    def list_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, title, created_at, updated_at FROM sessions ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        meta_json = json.dumps(metadata or {}, default=str)
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO messages (session_id, role, content, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
                (session_id, role, content, meta_json, now),
            )
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (now, session_id),
            )
        self._export_json(session_id)

    def log_run(
        self,
        session_id: str,
        query: str,
        intent: str,
        latency_ms: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO runs (session_id, query, intent, latency_ms, metadata, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    query,
                    intent,
                    latency_ms,
                    json.dumps(metadata or {}, default=str),
                    now,
                ),
            )

    def get_history(self, session_id: str, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT role, content, metadata, created_at FROM messages WHERE session_id = ? ORDER BY id ASC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        out = []
        for r in rows:
            item = dict(r)
            try:
                item["metadata"] = json.loads(item.get("metadata") or "{}")
            except json.JSONDecodeError:
                item["metadata"] = {}
            out.append(item)
        return out

    def load_session(self, session_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, title, created_at, updated_at FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
        if not row:
            return None
        return {
            **dict(row),
            "messages": self.get_history(session_id),
        }

    def _export_json(self, session_id: str) -> None:
        """Optional JSON snapshot for debugging."""
        export_dir = self.db_path.parent / "sessions"
        export_dir.mkdir(parents=True, exist_ok=True)
        data = self.load_session(session_id)
        if data:
            path = export_dir / f"{session_id}.json"
            path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
