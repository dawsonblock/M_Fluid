"""
Judge Memory Storage

SQLite storage for evidence, claims, and timeline.
Local-first, no external DB required.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from contextlib import contextmanager

from m_flow.judge_memory.config import JudgeMemoryConfig
from m_flow.judge_memory.models import EvidenceRecord, ClaimRecord, TimelineEvent
from m_flow.judge_memory.exceptions import StorageError


class JudgeMemoryStorage:
    """
    SQLite storage for Judge memory.

    Tables:
    - evidence_records: Immutable evidence storage
    - claim_records: Claims linked to evidence
    - timeline_events: Chronological events
    """

    def __init__(self, config: JudgeMemoryConfig):
        self.config = config
        self.db_path = config.sqlite_path
        self._ensure_dirs()
        self._init_db()

    def _ensure_dirs(self):
        """Ensure data directories exist."""
        self.config.data_dir.mkdir(parents=True, exist_ok=True)
        self.config.evidence_dir.mkdir(parents=True, exist_ok=True)

    def _init_db(self):
        """Initialize SQLite tables."""
        with self._connection() as conn:
            # Evidence records table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS evidence_records (
                    evidence_id TEXT PRIMARY KEY,
                    source_type TEXT NOT NULL,
                    source_url TEXT,
                    source_title TEXT,
                    retrieved_at TEXT NOT NULL,
                    published_at TEXT,
                    jurisdiction TEXT,
                    raw_text TEXT NOT NULL,
                    content_hash TEXT UNIQUE NOT NULL,
                    storage_path TEXT,
                    metadata_json TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Claim records table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS claim_records (
                    claim_id TEXT PRIMARY KEY,
                    evidence_id TEXT NOT NULL,
                    claim_text TEXT NOT NULL,
                    claim_type TEXT DEFAULT 'fact',
                    subject TEXT,
                    claim_status TEXT DEFAULT 'needs_verification',
                    confidence REAL DEFAULT 0.5,
                    jurisdiction TEXT,
                    case_id TEXT,
                    judge_id TEXT,
                    person_id TEXT,
                    event_date TEXT,
                    source_span_json TEXT,
                    metadata_json TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (evidence_id) REFERENCES evidence_records(evidence_id)
                )
            """)

            # Timeline events table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS timeline_events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    event_date TEXT,
                    title TEXT NOT NULL,
                    summary TEXT,
                    evidence_ids_json TEXT,
                    confidence REAL DEFAULT 0.5,
                    claim_status TEXT DEFAULT 'needs_verification',
                    metadata_json TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # FTS for text search (if available)
            try:
                conn.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS evidence_fts USING fts5(
                        evidence_id,
                        raw_text,
                        source_type,
                        content='evidence_records',
                        content_rowid='rowid'
                    )
                """)
            except sqlite3.OperationalError:
                # FTS5 not available, skip
                pass

            conn.commit()

    @contextmanager
    def _connection(self):
        """Get database connection."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _datetime_to_str(self, dt: Optional[datetime]) -> Optional[str]:
        """Convert datetime to ISO string."""
        if dt is None:
            return None
        return dt.isoformat()

    def _str_to_datetime(self, s: Optional[str]) -> Optional[datetime]:
        """Parse ISO string to datetime."""
        if s is None:
            return None
        return datetime.fromisoformat(s)

    # -------------------------------------------------------------------------
    # Evidence operations
    # -------------------------------------------------------------------------

    def store_evidence(self, record: EvidenceRecord) -> EvidenceRecord:
        """Store evidence record. Raises if duplicate hash."""
        with self._connection() as conn:
            # Check for duplicate hash
            existing = conn.execute(
                "SELECT evidence_id FROM evidence_records WHERE content_hash = ?",
                (record.content_hash,)
            ).fetchone()

            if existing:
                # Return existing record
                return self.get_evidence(existing["evidence_id"])

            # Insert new record
            conn.execute(
                """
                INSERT INTO evidence_records
                (evidence_id, source_type, source_url, source_title, retrieved_at,
                 published_at, jurisdiction, raw_text, content_hash, storage_path, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.evidence_id,
                    record.source_type,
                    record.source_url,
                    record.source_title,
                    self._datetime_to_str(record.retrieved_at),
                    self._datetime_to_str(record.published_at),
                    record.jurisdiction,
                    record.raw_text,
                    record.content_hash,
                    record.storage_path,
                    json.dumps(record.metadata) if record.metadata else None,
                )
            )
            conn.commit()
            return record

    def get_evidence(self, evidence_id: str) -> Optional[EvidenceRecord]:
        """Get evidence by ID."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM evidence_records WHERE evidence_id = ?",
                (evidence_id,)
            ).fetchone()

            if not row:
                return None

            return EvidenceRecord(
                evidence_id=row["evidence_id"],
                source_type=row["source_type"],
                source_url=row["source_url"],
                source_title=row["source_title"],
                retrieved_at=self._str_to_datetime(row["retrieved_at"]) or datetime.utcnow(),
                published_at=self._str_to_datetime(row["published_at"]),
                jurisdiction=row["jurisdiction"],
                raw_text=row["raw_text"],
                content_hash=row["content_hash"],
                storage_path=row["storage_path"],
                metadata=json.loads(row["metadata_json"]) if row["metadata_json"] else {},
            )

    def get_evidence_by_hash(self, content_hash: str) -> Optional[EvidenceRecord]:
        """Get evidence by content hash."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT evidence_id FROM evidence_records WHERE content_hash = ?",
                (content_hash,)
            ).fetchone()

            if row:
                return self.get_evidence(row["evidence_id"])
            return None

    def search_evidence(self, query: str, limit: int = 25) -> List[EvidenceRecord]:
        """Search evidence by text."""
        results = []
        with self._connection() as conn:
            # Simple LIKE search (FTS would be better but requires setup)
            pattern = f"%{query}%"
            rows = conn.execute(
                """
                SELECT evidence_id FROM evidence_records
                WHERE raw_text LIKE ? OR source_title LIKE ? OR source_url LIKE ?
                ORDER BY retrieved_at DESC
                LIMIT ?
                """,
                (pattern, pattern, pattern, limit)
            ).fetchall()

            for row in rows:
                record = self.get_evidence(row["evidence_id"])
                if record:
                    results.append(record)

        return results

    # -------------------------------------------------------------------------
    # Claim operations
    # -------------------------------------------------------------------------

    def store_claim(self, record: ClaimRecord) -> ClaimRecord:
        """Store claim record."""
        with self._connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO claim_records
                (claim_id, evidence_id, claim_text, claim_type, subject, claim_status,
                 confidence, jurisdiction, case_id, judge_id, person_id, event_date,
                 source_span_json, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.claim_id,
                    record.evidence_id,
                    record.claim_text,
                    record.claim_type,
                    record.subject,
                    record.claim_status,
                    record.confidence,
                    record.jurisdiction,
                    record.case_id,
                    record.judge_id,
                    record.person_id,
                    self._datetime_to_str(record.event_date),
                    json.dumps(record.source_span) if record.source_span else None,
                    json.dumps(record.metadata) if record.metadata else None,
                )
            )
            conn.commit()
            return record

    def get_claim(self, claim_id: str) -> Optional[ClaimRecord]:
        """Get claim by ID."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM claim_records WHERE claim_id = ?",
                (claim_id,)
            ).fetchone()

            if not row:
                return None

            return ClaimRecord(
                claim_id=row["claim_id"],
                evidence_id=row["evidence_id"],
                claim_text=row["claim_text"],
                claim_type=row["claim_type"] or "fact",
                subject=row["subject"],
                claim_status=row["claim_status"] or "needs_verification",
                confidence=row["confidence"] or 0.5,
                jurisdiction=row["jurisdiction"],
                case_id=row["case_id"],
                judge_id=row["judge_id"],
                person_id=row["person_id"],
                event_date=self._str_to_datetime(row["event_date"]),
                source_span=json.loads(row["source_span_json"]) if row["source_span_json"] else None,
                metadata=json.loads(row["metadata_json"]) if row["metadata_json"] else {},
            )

    def get_claims_by_evidence(self, evidence_id: str) -> List[ClaimRecord]:
        """Get all claims for an evidence record."""
        results = []
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT claim_id FROM claim_records WHERE evidence_id = ?",
                (evidence_id,)
            ).fetchall()

            for row in rows:
                claim = self.get_claim(row["claim_id"])
                if claim:
                    results.append(claim)

        return results

    def get_claims_by_case(self, case_id: str) -> List[ClaimRecord]:
        """Get all claims for a case."""
        results = []
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT claim_id FROM claim_records WHERE case_id = ?",
                (case_id,)
            ).fetchall()

            for row in rows:
                claim = self.get_claim(row["claim_id"])
                if claim:
                    results.append(claim)

        return results

    def get_claims_by_judge(self, judge_id: str) -> List[ClaimRecord]:
        """Get all claims for a judge."""
        results = []
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT claim_id FROM claim_records WHERE judge_id = ?",
                (judge_id,)
            ).fetchall()

            for row in rows:
                claim = self.get_claim(row["claim_id"])
                if claim:
                    results.append(claim)

        return results

    # -------------------------------------------------------------------------
    # Health check
    # -------------------------------------------------------------------------

    def healthcheck(self) -> Dict[str, Any]:
        """Check storage health."""
        try:
            with self._connection() as conn:
                evidence_count = conn.execute(
                    "SELECT COUNT(*) FROM evidence_records"
                ).fetchone()[0]
                claim_count = conn.execute(
                    "SELECT COUNT(*) FROM claim_records"
                ).fetchone()[0]

            return {
                "status": "ok",
                "sqlite_path": str(self.db_path),
                "evidence_count": evidence_count,
                "claim_count": claim_count,
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
            }
