"""SQLite storage: schema, connections, batched inserts and the queries used by the tools.

WAL mode lets one reader (the dashboard) and several writers (one per camera process)
work at the same time; busy_timeout makes a writer wait for the lock instead of failing.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import astuple, dataclass
from datetime import UTC, datetime
from pathlib import Path

from .config import Camera

SCHEMA = """
CREATE TABLE IF NOT EXISTS camera (
    camera_id TEXT PRIMARY KEY,
    name      TEXT NOT NULL,
    url       TEXT NOT NULL,
    line_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS crossing_event (
    event_id   TEXT PRIMARY KEY,
    camera_id  TEXT NOT NULL,
    ts_utc     TEXT NOT NULL,
    track_id   INTEGER NOT NULL,
    class_name TEXT NOT NULL,
    direction  TEXT NOT NULL CHECK (direction IN ('in', 'out')),
    confidence REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_event_cam_ts ON crossing_event (camera_id, ts_utc);
CREATE TABLE IF NOT EXISTS stream_health (
    camera_id TEXT NOT NULL,
    ts_utc    TEXT NOT NULL,
    status    TEXT NOT NULL,
    fps       REAL
);
CREATE INDEX IF NOT EXISTS idx_health_cam_ts ON stream_health (camera_id, ts_utc);
CREATE VIEW IF NOT EXISTS traffic_1min AS
    SELECT camera_id,
           substr(ts_utc, 1, 16) || ':00Z' AS minute_utc,
           class_name,
           direction,
           COUNT(*) AS n
    FROM crossing_event
    GROUP BY camera_id, minute_utc, class_name, direction;
"""


@dataclass(frozen=True)
class CrossingEvent:
    event_id: str
    camera_id: str
    ts_utc: str
    track_id: int
    class_name: str
    direction: str
    confidence: float


def utc_now() -> str:
    """Current UTC time as ISO-8601 with milliseconds, e.g. 2026-10-15T08:30:00.123Z."""
    return format_utc(datetime.now(UTC))


def format_utc(dt: datetime) -> str:
    """Same text format as the database, so string comparison == time comparison."""
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def make_event_id(camera_id: str, run_id: str, track_id: int, direction: str) -> str:
    """Deterministic id: the same crossing can never be stored twice."""
    key = f"{camera_id}|{run_id}|{track_id}|{direction}"
    return hashlib.sha1(key.encode()).hexdigest()


def connect(db_path: str | Path, read_only: bool = False) -> sqlite3.Connection:
    if read_only:
        conn = sqlite3.connect(f"file:{Path(db_path).resolve()}?mode=ro", uri=True, timeout=10)
    else:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


def upsert_camera(conn: sqlite3.Connection, cam: Camera) -> None:
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO camera (camera_id, name, url, line_json) VALUES (?, ?, ?, ?)",
            (cam.id, cam.name, cam.url, json.dumps(cam.line)),
        )


def insert_events(conn: sqlite3.Connection, events: list[CrossingEvent]) -> None:
    """All events in one transaction; duplicates (same event_id) are silently skipped."""
    with conn:
        conn.executemany(
            "INSERT OR IGNORE INTO crossing_event "
            "(event_id, camera_id, ts_utc, track_id, class_name, direction, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [astuple(e) for e in events],
        )


def insert_health(conn: sqlite3.Connection, camera_id: str, status: str, fps: float | None = None) -> None:
    with conn:
        conn.execute(
            "INSERT INTO stream_health (camera_id, ts_utc, status, fps) VALUES (?, ?, ?, ?)",
            (camera_id, utc_now(), status, fps),
        )


def count_by_class_direction(
    conn: sqlite3.Connection, camera_id: str, start_utc: str, end_utc: str
) -> dict[tuple[str, str], int]:
    """{(class_name, direction): count} for one camera in [start_utc, end_utc]."""
    rows = conn.execute(
        "SELECT class_name, direction, COUNT(*) FROM crossing_event "
        "WHERE camera_id = ? AND ts_utc BETWEEN ? AND ? GROUP BY class_name, direction",
        (camera_id, start_utc, end_utc),
    ).fetchall()
    return {(cls, d): n for cls, d, n in rows}


def minute_counts(conn: sqlite3.Connection, start_utc: str, end_utc: str) -> list[tuple]:
    """Same grouping as the traffic_1min view, but filtered on ts_utc so the index is used."""
    return conn.execute(
        "SELECT camera_id, substr(ts_utc, 1, 16) || ':00Z' AS minute_utc, class_name, direction, COUNT(*) "
        "FROM crossing_event WHERE ts_utc >= ? AND ts_utc < ? GROUP BY 1, 2, 3, 4",
        (start_utc, end_utc),
    ).fetchall()


def health_rows(conn: sqlite3.Connection, start_utc: str, end_utc: str) -> list[tuple]:
    """(camera_id, ts_utc, status, fps) in [start_utc, end_utc), plus the last row of each
    camera before the window, so a period that started earlier is not lost."""
    return conn.execute(
        "SELECT camera_id, ts_utc, status, fps FROM stream_health h "
        "WHERE (ts_utc >= ? AND ts_utc < ?) "
        "   OR ts_utc = (SELECT MAX(ts_utc) FROM stream_health p WHERE p.camera_id = h.camera_id AND p.ts_utc < ?) "
        "ORDER BY camera_id, ts_utc",
        (start_utc, end_utc, start_utc),
    ).fetchall()


def camera_names(conn: sqlite3.Connection) -> dict[str, str]:
    return dict(conn.execute("SELECT camera_id, name FROM camera ORDER BY camera_id").fetchall())
