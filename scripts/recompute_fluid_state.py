#!/usr/bin/env python3
"""Recompute fluid memory state for all evidence and claims in judge_memory.

Walks every evidence_id and claim_id in the database and calls
``FluidMemoryAdapter.touch_evidence()`` / ``touch_claim()`` to
refresh activation, salience, and decay values stored in the
standalone ``fluid_memory`` engine.

Requires ``enable_fluid_memory=True`` in judgement_memory config
(or pass ``--force`` to proceed even when the adapter reports disabled).

Usage:
    python scripts/recompute_fluid_state.py [--data-dir PATH] [--force] [--dry-run]
"""

import argparse
import asyncio
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


async def recompute(db_path: Path, dry_run: bool, force: bool) -> None:
    try:
        from judge_memory import JudgeMemoryConfig, JudgeMemoryService
    except ImportError as exc:
        print(f"Cannot import judge_memory: {exc}", file=sys.stderr)
        sys.exit(1)

    config = JudgeMemoryConfig(
        data_dir=str(db_path.parent),
        enable_fluid_memory=True,
    )
    service = JudgeMemoryService(config)

    if not service.fluid or not service.fluid.enabled:
        if force:
            print(
                "WARNING: fluid adapter is disabled; --force flag set, continuing anyway."
            )
        else:
            print(
                "Fluid memory is not enabled. "
                "Pass --force to proceed anyway, or set enable_fluid_memory=True in your config."
            )
            sys.exit(1)

    conn = sqlite3.connect(str(db_path))

    evidence_rows = conn.execute(
        "SELECT evidence_id, source_type FROM evidence ORDER BY created_at"
    ).fetchall()
    claim_rows = conn.execute(
        "SELECT claim_id, evidence_id FROM claims ORDER BY created_at"
    ).fetchall()
    conn.close()

    print(f"Evidence records: {len(evidence_rows)}")
    print(f"Claim records:    {len(claim_rows)}")
    if dry_run:
        print("Dry-run mode — no state will be written.")
        return

    ev_ok = ev_err = 0
    for evidence_id, source_type in evidence_rows:
        try:
            await service.fluid.touch_evidence(evidence_id, source_type or "unknown")
            ev_ok += 1
        except Exception as exc:
            print(f"  WARN evidence {evidence_id}: {exc}")
            ev_err += 1

    cl_ok = cl_err = 0
    for claim_id, evidence_id in claim_rows:
        source_type = "unknown"
        try:
            source_type = conn.execute(
                "SELECT source_type FROM evidence WHERE evidence_id=?", (evidence_id,)
            ).fetchone()
            source_type = source_type[0] if source_type else "unknown"
        except Exception:
            pass
        try:
            conn2 = sqlite3.connect(str(db_path))
            row = conn2.execute(
                "SELECT source_type FROM evidence WHERE evidence_id=?", (evidence_id,)
            ).fetchone()
            conn2.close()
            source_type = row[0] if row else "unknown"
            await service.fluid.touch_claim(claim_id, source_type)
            cl_ok += 1
        except Exception as exc:
            print(f"  WARN claim {claim_id}: {exc}")
            cl_err += 1

    print(f"\nEvidence: {ev_ok} updated, {ev_err} error(s)")
    print(f"Claims:   {cl_ok} updated, {cl_err} error(s)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default="./judge_memory_data",
        help="Path to the judge_memory data directory (default: ./judge_memory_data)",
    )
    parser.add_argument("--db", default=None, help="Path to a specific .db file")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without writing any state",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Continue even if fluid adapter reports it is disabled",
    )
    args = parser.parse_args()

    db_path = Path(args.db) if args.db else _find_db(Path(args.data_dir))
    if not db_path.exists():
        print(f"Database not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Database: {db_path}")
    asyncio.run(recompute(db_path, dry_run=args.dry_run, force=args.force))


if __name__ == "__main__":
    main()
