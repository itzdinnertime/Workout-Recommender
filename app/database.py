"""
SQLite database layer for workout logs.

Schema: one row per set logged. Kept intentionally simple (flat table,
no ORM) so the recommendation logic can be understood without wading
through abstraction layers.
"""
import sqlite3
from contextlib import contextmanager
from datetime import date
from typing import Optional

DB_PATH = "workouts.db"


def _resolve(db_path: Optional[str]) -> str:
    """
    Resolves the effective db path at CALL time, not at function-definition
    time. This matters: `def f(db_path=DB_PATH)` captures DB_PATH's value
    once, when the module is first imported -- so monkeypatching
    `database.DB_PATH` later (e.g. in tests) silently has no effect on
    calls using the default. Routing every function through this helper
    fixes that, so tests can safely redirect all DB access to a temp file.
    """
    return db_path if db_path is not None else DB_PATH


def init_db(db_path: Optional[str] = None):
    db_path = _resolve(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS workout_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            exercise TEXT NOT NULL,
            exercise_type TEXT NOT NULL,  -- 'upper' or 'lower', affects increment size
            log_date TEXT NOT NULL,       -- ISO date string
            weight REAL NOT NULL,
            reps_target INTEGER NOT NULL,
            reps_achieved INTEGER NOT NULL,
            sets INTEGER NOT NULL DEFAULT 1
        )
    """)
    conn.commit()
    conn.close()


@contextmanager
def get_conn(db_path: Optional[str] = None):
    db_path = _resolve(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def insert_log(user_id: str, exercise: str, exercise_type: str, log_date: str,
               weight: float, reps_target: int, reps_achieved: int, sets: int = 1,
               db_path: Optional[str] = None):
    with get_conn(db_path) as conn:
        conn.execute(
            """INSERT INTO workout_logs
               (user_id, exercise, exercise_type, log_date, weight, reps_target, reps_achieved, sets)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (user_id, exercise, exercise_type, log_date, weight, reps_target, reps_achieved, sets)
        )


def get_history(user_id: str, exercise: str, db_path: Optional[str] = None, limit: Optional[int] = None):
    with get_conn(db_path) as conn:
        query = """SELECT * FROM workout_logs
                   WHERE user_id = ? AND exercise = ?
                   ORDER BY log_date DESC"""
        if limit:
            query += f" LIMIT {int(limit)}"
        rows = conn.execute(query, (user_id, exercise)).fetchall()
        return [dict(r) for r in rows]
