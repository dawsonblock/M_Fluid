#!/usr/bin/env python3
"""Rebuild all SQLite FTS5 full-text-search indexes for judge_memory.

Usage:
    python scripts/rebuild_memory_indexes.py [--data-dir PATH]

This is safe to run on a live database — SQLite serialises the rebuild
inside a transaction and the table is still readable throughout.
"""

import argparse
import sqlite3
import sys
import time
from pathlib import Path


def _find_db(data_dir: Path) -> Path:
    candidates = list(data_dir.glob("*.db")) + list(data_dir.glob("**/*.db"))
    if not candidates:
        print(f"No .db files found under {data_dir}", file=sys.stderr)
        sys.exit(1)
    if len(candidates) == 1:
        return candidates[0]
    # Prefer judge_memory.db if present
    for c in candidates:
        if "judge_memory" in c.name:
            return c
    return candidates[0]


def rebuild_fts5_table(conn: sqlite3.Connection, fts_table: str) -> None:
    """Issue INSERT INTO <table>(<table>) VALUES('rebuild') to rebuild."""
    conn.execute(f"INSERT INTO {fts_table}({fts_table}) VALUES('rebuild')")


def list_fts5_tables(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    fts = []
    for (name,) in rows:
        try:
            conn.execute(f"SELECT * FROM {name} WHERE {name} MATCH 'x' LIMIT 0")
            fts.append(name)
        except sqlite3.OperationalError:
            pass
    return fts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default="./judge_memory_data",
        help="Path to the judge_memory data directory (default: ./judge_memory_data)",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Path to a specific .db file (overrides --data-dir auto-detection)",
    )
    args = parser.parse_args()

    db_path = Path(args.db) if args.db else _find_db(Path(args.data_dir))
    if not db_path.exists():
        print(f"Database not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Database: {db_path}")
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    fts_tables = list_fts5_tables(conn)
    if not fts_tables:
        print("No FTS5 tables found — nothing to rebuild.")
        conn.close()
        return

    print(f"FTS5 tables found: {fts_tables}")
    for table in fts_tables:
        t0 = time.monotonic()
        try:
            with conn:
                rebuild_fts5_table(conn, table)
            elapsed = time.monotonic() - t0
            print(f"  ✓ {table} rebuilt in {elapsed:.2f}s")
        except sqlite3.OperationalError as exc:
            print(f"  ✗ {table} failed: {exc}", file=sys.stderr)

    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
