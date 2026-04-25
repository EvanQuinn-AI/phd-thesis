"""Schema migrations. Versioned, idempotent."""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime


_HERE = os.path.dirname(os.path.abspath(__file__))


def _read_schema_sql() -> str:
    with open(os.path.join(_HERE, "schema.sql")) as f:
        return f.read()


# (version, sql) pairs. Append-only; never edit historical entries.
MIGRATIONS: list[tuple[int, str]] = [
    (1, _read_schema_sql()),
]


def _current_version(conn: sqlite3.Connection) -> int:
    try:
        row = conn.execute(
            "SELECT MAX(version) FROM schema_version"
        ).fetchone()
        return int(row[0] or 0)
    except sqlite3.OperationalError:
        # schema_version table doesn't exist yet
        return 0


def run_migrations(conn: sqlite3.Connection) -> None:
    current = _current_version(conn)
    for version, sql in MIGRATIONS:
        if version <= current:
            continue
        conn.executescript(sql)
        conn.execute(
            "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
            (version, datetime.utcnow().isoformat()),
        )
    conn.commit()
