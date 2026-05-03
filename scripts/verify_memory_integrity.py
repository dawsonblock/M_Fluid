#!/usr/bin/env python3
"""Verify integrity of the judge_memory SQLite database.

Checks:
  1. All evidence files referenced in the DB exist on disk.
  2. SHA256 hashes match the stored content_hash values.
  3. All claims reference an existing evidence_id.
  4. FTS5 index row-count matches the source tables.

Usage:
    python scripts/verify_memory_integrity.py [--data-dir PATH] [--fix-missing]

Exit codes:
    0  all checks passed
    1  one or more integrity violations found
"""

import argparse
import hashlib
import sqlite3
import sys
from pathlib import Path


def _find_db(data_dir: Path) -> Path:
    candidates = list(data_dir.glob("*.db")) + list(data_dir.glob("**/*.db"))
    if not candidates:
        print(f"No .db files found under {data_dir}", file=sys.stderr)
        sys.exit(1)
    for c in candidates:
        if "judge_memory" in c.name:
            return c
    return candidates[0]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def check_evidence_files(conn: sqlite3.Connection, data_dir: Path) -> list[str]:
    """Verify every evidence record has a file and the hash matches."""
    errors: list[str] = []
    rows = conn.execute(
        "SELECT evidence_id, content_hash, evidence_path FROM evidence"
    ).fetchall()
    for evidence_id, content_hash, evidence_path in rows:
        if evidence_path is None:
            continue
        path = Path(evidence_path)
        if not path.is_absolute():
            path = data_dir / path
        if not path.exists():
            errors.append(f"MISSING FILE  evidence_id={evidence_id}  path={path}")
            continue
        if content_hash:
            actual = sha256_file(path)
            if actual != content_hash:
                errors.append(
                    f"HASH MISMATCH evidence_id={evidence_id}  "
                    f"expected={content_hash[:12]}...  actual={actual[:12]}..."
                )
    return errors


def check_orphan_claims(conn: sqlite3.Connection) -> list[str]:
    """Claims must reference an existing evidence row."""
    errors: list[str] = []
    rows = conn.execute("""
        SELECT c.claim_id, c.evidence_id
        FROM claims c
        LEFT JOIN evidence e ON e.evidence_id = c.evidence_id
        WHERE e.evidence_id IS NULL
        """).fetchall()
    for claim_id, evidence_id in rows:
        errors.append(
            f"ORPHAN CLAIM  claim_id={claim_id}  missing evidence_id={evidence_id}"
        )
    return errors


def check_fts_counts(conn: sqlite3.Connection) -> list[str]:
    """FTS5 virtual table row-count should match its content table."""
    errors: list[str] = []
    pairs = [("evidence_fts", "evidence"), ("claims_fts", "claims")]
    tables_present = {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    for fts_table, content_table in pairs:
        if fts_table not in tables_present or content_table not in tables_present:
            continue
        fts_count = conn.execute(f"SELECT count(*) FROM {fts_table}").fetchone()[0]
        src_count = conn.execute(f"SELECT count(*) FROM {content_table}").fetchone()[0]
        if fts_count != src_count:
            errors.append(
                f"FTS COUNT MISMATCH  {fts_table}={fts_count}  {content_table}={src_count}  "
                f"(run rebuild_memory_indexes.py to fix)"
            )
    return errors


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
        help="Path to a specific .db file",
    )
    args = parser.parse_args()

    db_path = Path(args.db) if args.db else _find_db(Path(args.data_dir))
    if not db_path.exists():
        print(f"Database not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    data_dir = db_path.parent
    print(f"Database: {db_path}")

    conn = sqlite3.connect(str(db_path))
    all_errors: list[str] = []

    print("Checking evidence file integrity...")
    errors = check_evidence_files(conn, data_dir)
    all_errors.extend(errors)
    print(f"  {len(errors)} issue(s)")

    print("Checking claim references...")
    errors = check_orphan_claims(conn)
    all_errors.extend(errors)
    print(f"  {len(errors)} issue(s)")

    print("Checking FTS index counts...")
    errors = check_fts_counts(conn)
    all_errors.extend(errors)
    print(f"  {len(errors)} issue(s)")

    conn.close()

    if all_errors:
        print(f"\n{len(all_errors)} integrity violation(s) found:")
        for err in all_errors:
            print(f"  {err}")
        sys.exit(1)
    else:
        print("\nAll integrity checks passed.")


if __name__ == "__main__":
    main()
