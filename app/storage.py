"""Small local SQLite store for ADAM conversation history and tasks."""

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class LocalStore:
    def __init__(self, database_path: Optional[Path] = None, db_path: Optional[Path] = None):
        target = database_path or db_path
        self.database_path = target or (Path(__file__).resolve().parent.parent / "data" / "adam.db")
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversation (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT NOT NULL CHECK (role IN ('USER', 'ADAM')),
                    source TEXT NOT NULL CHECK (source IN ('text', 'voice', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    completed INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS personal_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT NOT NULL UNIQUE,
                    value TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT 'fact',
                    source TEXT NOT NULL DEFAULT 'user_explicit',
                    confidence REAL NOT NULL DEFAULT 1.0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_personal_memory_key ON personal_memory(key);
                CREATE INDEX IF NOT EXISTS idx_personal_memory_cat ON personal_memory(category);
                """
            )
            task_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(tasks)").fetchall()
            }
            if "request_id" not in task_columns:
                connection.execute("ALTER TABLE tasks ADD COLUMN request_id TEXT")
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS tasks_request_id_unique "
                "ON tasks(request_id) WHERE request_id IS NOT NULL"
            )
        try:
            os.chmod(self.database_path, 0o600)
        except OSError:
            pass

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.database_path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 5000")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def add_message(self, role: str, content: str, source: str = "text") -> None:
        role = role.upper()
        if role not in ("USER", "ADAM") or not content.strip():
            return
        if role == "ADAM":
            source = "assistant"
        elif source not in ("text", "voice"):
            source = "text"
        created_at = datetime.now().astimezone().isoformat(timespec="seconds")
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO conversation (role, source, content, created_at) VALUES (?, ?, ?, ?)",
                (role, source, content, created_at),
            )

    def list_messages(self, limit: int = 500) -> List[Dict[str, str]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT role, source, content, created_at FROM conversation ORDER BY id DESC LIMIT ?",
                (max(1, min(limit, 5000)),),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def clear_history(self) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM conversation")

    def add_task(self, title: str, request_id: Optional[str] = None) -> int:
        title = " ".join(title.split())
        if not title:
            raise ValueError("Task title cannot be empty.")
        if len(title) > 200:
            raise ValueError("Task title cannot exceed 200 characters.")
        if request_id is not None:
            request_id = request_id.strip()
            if not request_id or len(request_id) > 128:
                raise ValueError("Task request ID must contain 1 to 128 characters.")
        created_at = datetime.now().astimezone().isoformat(timespec="seconds")
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT OR IGNORE INTO tasks (title, created_at, request_id) VALUES (?, ?, ?)",
                (title, created_at, request_id),
            )
            if cursor.rowcount:
                task_id = cursor.lastrowid
            else:
                row = connection.execute(
                    "SELECT id FROM tasks WHERE request_id = ?",
                    (request_id,),
                ).fetchone()
                if row is None:
                    raise RuntimeError("Task was not saved and no matching request was found.")
                task_id = row["id"]
            return int(task_id)

    def list_tasks(self) -> List[Dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, title, completed, created_at FROM tasks ORDER BY completed, id DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def set_task_completed(self, task_id: int, completed: bool) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE tasks SET completed = ? WHERE id = ?",
                (int(completed), task_id),
            )

    def delete_task(self, task_id: int) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM tasks WHERE id = ?", (task_id,))

    # =========================================================================
    # PERSONAL MEMORY STORE
    # =========================================================================

    def get_memory(self, key: str) -> Optional[Dict[str, object]]:
        """Retrieve a specific stored personal memory fact or preference by key."""
        clean_key = str(key).strip().lower()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, key, value, category, source, confidence, created_at, updated_at "
                "FROM personal_memory WHERE key = ?",
                (clean_key,),
            ).fetchone()
        return dict(row) if row else None

    def set_memory(
        self,
        key: str,
        value: str,
        category: str = "fact",
        source: str = "user_explicit",
        confidence: float = 1.0,
    ) -> Dict[str, object]:
        """Store or update a personal fact or preference."""
        clean_key = str(key).strip().lower()
        clean_val = str(value).strip()
        now = datetime.now().astimezone().isoformat(timespec="seconds")

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO personal_memory (key, value, category, source, confidence, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    category = excluded.category,
                    source = excluded.source,
                    confidence = excluded.confidence,
                    updated_at = excluded.updated_at
                """,
                (clean_key, clean_val, category, source, float(confidence), now, now),
            )
            row = connection.execute(
                "SELECT id, key, value, category, source, confidence, created_at, updated_at "
                "FROM personal_memory WHERE key = ?",
                (clean_key,),
            ).fetchone()
        return dict(row) if row else {}

    def delete_memory(self, key: str) -> bool:
        """Remove a specific memory by key."""
        clean_key = str(key).strip().lower()
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM personal_memory WHERE key = ?", (clean_key,)
            )
            return cursor.rowcount > 0

    def list_memories(self, category: Optional[str] = None) -> List[Dict[str, object]]:
        """List all stored personal memories, optionally filtered by category."""
        with self._connect() as connection:
            if category:
                rows = connection.execute(
                    "SELECT id, key, value, category, source, confidence, created_at, updated_at "
                    "FROM personal_memory WHERE category = ? ORDER BY key ASC",
                    (str(category).strip().lower(),),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT id, key, value, category, source, confidence, created_at, updated_at "
                    "FROM personal_memory ORDER BY category ASC, key ASC"
                ).fetchall()
        return [dict(row) for row in rows]

    def clear_memories(self) -> None:
        """Clear all personal memories."""
        with self._connect() as connection:
            connection.execute("DELETE FROM personal_memory")
