"""Judge Memory SQLite Storage"""

import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from judge_memory.config import JudgeMemoryConfig
from judge_memory.models import EvidenceRecord, ClaimRecord, TimelineEvent
from judge_memory.exceptions import StorageError, EvidenceNotFoundError
from judge_memory._logger import get_logger

logger = get_logger(__name__)


def _sanitize_fts_query(query: str) -> str:
    """Sanitize FTS5 MATCH query to prevent syntax errors.

    Escapes special characters and removes FTS5 operators that could
    cause query failures or unexpected behavior. Also removes operator
    words (OR, AND, NOT, NEAR) to ensure literal token matching.

    Args:
        query: Raw user query string

    Returns:
        Sanitized query safe for FTS5 MATCH (tokens joined with AND)
    """
    if not query:
        return ""

    # Remove or escape FTS5 special characters
    # FTS5 special chars: " * ( ) - + : ~ . / \ @ # $ % ^ & = < >
    sanitized = re.sub(r'["*()\-+:~./\\@#$%^&=<>]', ' ', query)

    # Collapse multiple spaces
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()

    # If query is empty after sanitization, return empty
    if not sanitized:
        return ""

    # Split into tokens and remove FTS5 operator words
    tokens = sanitized.split()
    operator_words = {'OR', 'AND', 'NOT', 'NEAR'}
    tokens = [t for t in tokens if t.upper() not in operator_words]

    # Return space-separated tokens (FTS5 treats spaces as implicit AND)
    if tokens:
        return ' '.join(tokens)
    return ""


class JudgeMemoryStorage:
    """SQLite storage for Judge Memory.
    
    Self-contained storage with no external dependencies.
    Uses JSON serialization for complex fields.
    """
    
    def __init__(self, config: JudgeMemoryConfig):
        self.config = config
        self.db_path = config.db_path
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize database tables."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            # Enable foreign key constraints
            conn.execute("PRAGMA foreign_keys=ON")
            
            # Evidence table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS evidence_records (
                    evidence_id TEXT PRIMARY KEY,
                    content_hash TEXT UNIQUE NOT NULL,
                    source_type TEXT NOT NULL,
                    source_url TEXT,
                    source_title TEXT,
                    content_preview TEXT,
                    jurisdiction TEXT,
                    published_at TEXT,
                    file_path TEXT,
                    metadata TEXT,
                    ingested_at TEXT NOT NULL
                )
            """)
            
            # Claims table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS claim_records (
                    claim_id TEXT PRIMARY KEY,
                    evidence_id TEXT NOT NULL,
                    claim_text TEXT NOT NULL,
                    claim_type TEXT,
                    case_id TEXT,
                    judge_id TEXT,
                    person_id TEXT,
                    entity_id TEXT,
                    confidence REAL DEFAULT 0.5,
                    status TEXT DEFAULT 'active',
                    tags TEXT,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (evidence_id) REFERENCES evidence_records(evidence_id)
                )
            """)
            
            # Timeline events table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS timeline_events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    event_date TEXT NOT NULL,
                    description TEXT NOT NULL,
                    evidence_id TEXT,
                    claim_id TEXT,
                    case_id TEXT,
                    judge_id TEXT,
                    person_id TEXT,
                    entity_id TEXT,
                    jurisdiction TEXT,
                    metadata TEXT
                )
            """)
            
            # Create indexes for common queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_evidence_hash 
                ON evidence_records(content_hash)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_claims_evidence 
                ON claim_records(evidence_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_claims_case 
                ON claim_records(case_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_claims_judge 
                ON claim_records(judge_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_claims_entity 
                ON claim_records(entity_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timeline_case 
                ON timeline_events(case_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timeline_judge 
                ON timeline_events(judge_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timeline_entity 
                ON timeline_events(entity_id)
            """)

            # FTS5 virtual tables for full-text search (optional, gracefully degrades)
            try:
                # Standalone FTS tables (not external-content) with id column for correlation
                conn.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS evidence_fts USING fts5(
                        evidence_id,
                        content_preview,
                        source_title,
                        source_url
                    )
                """)
                conn.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS claims_fts USING fts5(
                        claim_id,
                        claim_text
                    )
                """)
                logger.debug("FTS5 tables initialized")

                # Create triggers for automatic FTS maintenance
                self._create_fts_triggers(conn)

                # Auto-backfill FTS if tables are empty but source tables have data
                self._maybe_backfill_fts(conn)

            except sqlite3.OperationalError as e:
                logger.warning(f"FTS5 not available ({e}), falling back to LIKE search")

            conn.commit()
            logger.info(f"Database initialized: {self.db_path}")

    def _create_fts_triggers(self, conn: sqlite3.Connection) -> None:
        """Create triggers to automatically maintain FTS indexes (standalone tables)."""
        # Evidence triggers - use DELETE+INSERT pattern for reliable updates
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS evidence_fts_insert AFTER INSERT ON evidence_records BEGIN
                INSERT INTO evidence_fts(evidence_id, content_preview, source_title, source_url)
                VALUES (new.evidence_id, new.content_preview, new.source_title, new.source_url);
            END
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS evidence_fts_update AFTER UPDATE ON evidence_records BEGIN
                DELETE FROM evidence_fts WHERE evidence_id = old.evidence_id;
                INSERT INTO evidence_fts(evidence_id, content_preview, source_title, source_url)
                VALUES (new.evidence_id, new.content_preview, new.source_title, new.source_url);
            END
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS evidence_fts_delete AFTER DELETE ON evidence_records BEGIN
                DELETE FROM evidence_fts WHERE evidence_id = old.evidence_id;
            END
        """)

        # Claims triggers - use DELETE+INSERT pattern for reliable updates
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS claims_fts_insert AFTER INSERT ON claim_records BEGIN
                INSERT INTO claims_fts(claim_id, claim_text)
                VALUES (new.claim_id, new.claim_text);
            END
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS claims_fts_update AFTER UPDATE ON claim_records BEGIN
                DELETE FROM claims_fts WHERE claim_id = old.claim_id;
                INSERT INTO claims_fts(claim_id, claim_text)
                VALUES (new.claim_id, new.claim_text);
            END
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS claims_fts_delete AFTER DELETE ON claim_records BEGIN
                DELETE FROM claims_fts WHERE claim_id = old.claim_id;
            END
        """)
        logger.debug("FTS triggers created")

    def _maybe_backfill_fts(self, conn: sqlite3.Connection) -> None:
        """Auto-backfill FTS indexes if they're empty but source tables have data."""
        try:
            # Check if evidence_fts is empty
            fts_count = conn.execute("SELECT COUNT(*) FROM evidence_fts").fetchone()[0]
            source_count = conn.execute("SELECT COUNT(*) FROM evidence_records").fetchone()[0]

            if fts_count == 0 and source_count > 0:
                logger.info("FTS tables empty but %d evidence records exist. Backfilling...", source_count)
                self._rebuild_fts_indexes_internal(conn)
        except sqlite3.OperationalError:
            # FTS5 not available, skip
            pass

    def _rebuild_fts_indexes_internal(self, conn: sqlite3.Connection) -> None:
        """Internal method to rebuild FTS indexes using an existing connection."""
        # Clear existing FTS content
        conn.execute("DELETE FROM evidence_fts")
        conn.execute("DELETE FROM claims_fts")

        # Rebuild evidence FTS (standalone table with evidence_id column)
        conn.execute("""
            INSERT INTO evidence_fts(evidence_id, content_preview, source_title, source_url)
            SELECT evidence_id, content_preview, source_title, source_url
            FROM evidence_records
        """)

        # Rebuild claims FTS (standalone table with claim_id column)
        conn.execute("""
            INSERT INTO claims_fts(claim_id, claim_text)
            SELECT claim_id, claim_text
            FROM claim_records
        """)

        logger.info("FTS indexes rebuilt successfully")

    def rebuild_fts_indexes(self) -> Dict[str, int]:
        """Manually rebuild FTS indexes from source tables.

        Returns:
            Dict with evidence_count and claims_count of indexed records.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                self._rebuild_fts_indexes_internal(conn)

                evidence_count = conn.execute("SELECT COUNT(*) FROM evidence_fts").fetchone()[0]
                claims_count = conn.execute("SELECT COUNT(*) FROM claims_fts").fetchone()[0]

                conn.commit()

                return {
                    "evidence_count": evidence_count,
                    "claims_count": claims_count,
                }
        except sqlite3.OperationalError as e:
            logger.error(f"Failed to rebuild FTS indexes: {e}")
            raise StorageError(f"FTS rebuild failed: {e}")
    
    def store_evidence(self, record: EvidenceRecord) -> EvidenceRecord:
        """Store evidence record.

        Evidence is content-addressed. If the content hash already exists,
        return the existing evidence record instead of replacing it with a
        new evidence_id.
        """
        try:
            existing = self.get_evidence_by_hash(record.content_hash)
            if existing:
                return existing

            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO evidence_records (
                        evidence_id, content_hash, source_type, source_url,
                        source_title, content_preview, jurisdiction, published_at, file_path,
                        metadata, ingested_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.evidence_id,
                        record.content_hash,
                        record.source_type,
                        record.source_url,
                        record.source_title,
                        record.content_preview,
                        record.jurisdiction,
                        record.published_at.isoformat() if record.published_at else None,
                        record.file_path,
                        json.dumps(record.metadata),
                        record.ingested_at.isoformat(),
                    ),
                )
                conn.commit()
                return record
        except sqlite3.Error as e:
            logger.error(f"Failed to store evidence: {e}")
            raise StorageError(f"Failed to store evidence: {e}")
    
    def get_evidence(self, evidence_id: str) -> Optional[EvidenceRecord]:
        """Retrieve evidence by ID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT * FROM evidence_records WHERE evidence_id = ?",
                    (evidence_id,),
                ).fetchone()
                
                if not row:
                    return None
                    
                return self._row_to_evidence(row)
        except sqlite3.Error as e:
            logger.error(f"Failed to get evidence: {e}")
            raise StorageError(f"Failed to get evidence: {e}")
    
    def get_evidence_by_hash(self, content_hash: str) -> Optional[EvidenceRecord]:
        """Retrieve evidence by content hash."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT * FROM evidence_records WHERE content_hash = ?",
                    (content_hash,),
                ).fetchone()
                
                if not row:
                    return None
                    
                return self._row_to_evidence(row)
        except sqlite3.Error as e:
            logger.error(f"Failed to get evidence by hash: {e}")
            raise StorageError(f"Failed to get evidence by hash: {e}")
    
    def search_evidence(
        self,
        query: Optional[str] = None,
        source_type: Optional[str] = None,
        jurisdiction: Optional[str] = None,
        limit: int = 50,
    ) -> List[EvidenceRecord]:
        """Search evidence records."""
        try:
            sql = "SELECT * FROM evidence_records WHERE 1=1"
            params = []
            
            if query:
                sql += " AND (source_title LIKE ? OR source_url LIKE ? OR content_preview LIKE ?)"
                params.extend([f"%{query}%", f"%{query}%", f"%{query}%"])
            
            if source_type:
                sql += " AND source_type = ?"
                params.append(source_type)
            
            if jurisdiction:
                sql += " AND jurisdiction = ?"
                params.append(jurisdiction)
            
            safe_limit = self._clamp_limit(limit)
            sql += " ORDER BY ingested_at DESC LIMIT ?"
            params.append(safe_limit)

            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(sql, params).fetchall()
                return [self._row_to_evidence(row) for row in rows]
                
        except sqlite3.Error as e:
            logger.error(f"Failed to search evidence: {e}")
            raise StorageError(f"Failed to search evidence: {e}")

    def search_evidence_fts(
        self,
        query: str,
        source_type: Optional[str] = None,
        jurisdiction: Optional[str] = None,
        limit: int = 50,
    ) -> List[tuple[EvidenceRecord, float]]:
        """Search evidence using FTS5 with BM25 ranking.

        Returns:
            List of (evidence_record, bm25_score) tuples, sorted by score descending.
        """
        try:
            safe_limit = self._clamp_limit(limit)

            with sqlite3.connect(self.db_path) as conn:
                # Check if FTS5 is available
                try:
                    conn.execute("SELECT * FROM evidence_fts LIMIT 1")
                except sqlite3.OperationalError:
                    # FTS5 not available, fall back to LIKE search with dummy scores
                    records = self.search_evidence(query, source_type, jurisdiction, limit)
                    return [(r, 0.5) for r in records]

                # Sanitize query for FTS5 safety
                safe_query = _sanitize_fts_query(query)
                if not safe_query:
                    return []

                # Build FTS5 query with filters
                # Join evidence_records on evidence_id, rank by bm25
                sql = """
                    SELECT e.*, rank FROM evidence_fts f
                    JOIN evidence_records e ON e.evidence_id = f.evidence_id
                    WHERE evidence_fts MATCH ?
                """
                params = [safe_query]

                if source_type:
                    sql += " AND e.source_type = ?"
                    params.append(source_type)

                if jurisdiction:
                    sql += " AND e.jurisdiction = ?"
                    params.append(jurisdiction)

                sql += " ORDER BY rank LIMIT ?"
                params.append(safe_limit)

                rows = conn.execute(sql, params).fetchall()
                results = []
                for row in rows:
                    # Last column is rank (bm25 score, lower is better for FTS5)
                    # Convert to normalized score (0-1, higher is better)
                    bm25_rank = row[-1]
                    score = max(0.0, min(1.0, 1.0 / (1.0 + abs(bm25_rank))))
                    evidence = self._row_to_evidence(row[:-1])
                    results.append((evidence, score))

                return results

        except sqlite3.Error as e:
            logger.error(f"Failed FTS5 evidence search: {e}")
            # Fallback to regular search
            records = self.search_evidence(query, source_type, jurisdiction, limit)
            return [(r, 0.5) for r in records]

    def search_claims_fts(
        self,
        query: str,
        case_id: Optional[str] = None,
        judge_id: Optional[str] = None,
        person_id: Optional[str] = None,
        entity_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> List[tuple[ClaimRecord, float]]:
        """Search claims using FTS5 with BM25 ranking.

        Returns:
            List of (claim_record, bm25_score) tuples, sorted by score descending.
        """
        try:
            safe_limit = self._clamp_limit(limit)

            with sqlite3.connect(self.db_path) as conn:
                # Check if FTS5 is available
                try:
                    conn.execute("SELECT * FROM claims_fts LIMIT 1")
                except sqlite3.OperationalError:
                    # FTS5 not available, fall back to LIKE search
                    records = self.search_claims(query, case_id, judge_id, person_id, entity_id, status, limit)
                    return [(r, 0.5) for r in records]

                # Sanitize query for FTS5 safety
                safe_query = _sanitize_fts_query(query)
                if not safe_query:
                    return []

                # Build FTS5 query with filters
                sql = """
                    SELECT c.*, rank FROM claims_fts f
                    JOIN claim_records c ON c.claim_id = f.claim_id
                    WHERE claims_fts MATCH ?
                """
                params = [safe_query]

                if case_id:
                    sql += " AND c.case_id = ?"
                    params.append(case_id)
                if judge_id:
                    sql += " AND c.judge_id = ?"
                    params.append(judge_id)
                if person_id:
                    sql += " AND c.person_id = ?"
                    params.append(person_id)
                if entity_id:
                    sql += " AND c.entity_id = ?"
                    params.append(entity_id)
                if status:
                    sql += " AND c.status = ?"
                    params.append(status)

                sql += " ORDER BY rank LIMIT ?"
                params.append(safe_limit)

                rows = conn.execute(sql, params).fetchall()
                results = []
                for row in rows:
                    bm25_rank = row[-1]
                    score = max(0.0, min(1.0, 1.0 / (1.0 + abs(bm25_rank))))
                    claim = self._row_to_claim(row[:-1])
                    results.append((claim, score))

                return results

        except sqlite3.Error as e:
            logger.error(f"Failed FTS5 claims search: {e}")
            # Fallback to regular search
            records = self.search_claims(query, case_id, judge_id, person_id, entity_id, status, limit)
            return [(r, 0.5) for r in records]

    def search_claims_with_evidence(
        self,
        query: str,
        case_id: Optional[str] = None,
        judge_id: Optional[str] = None,
        person_id: Optional[str] = None,
        entity_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
        use_fts: bool = True,
    ) -> List[Tuple[ClaimRecord, EvidenceRecord, float]]:
        """Search claims joined with their evidence records for full metadata.

        Returns claims with their parent evidence so source metadata
        (source_type, jurisdiction, source_title, etc.) is available.

        Returns:
            List of (claim_record, evidence_record, score) tuples
        """
        try:
            safe_limit = self._clamp_limit(limit)

            with sqlite3.connect(self.db_path) as conn:
                if use_fts:
                    # Check if FTS5 is available
                    try:
                        conn.execute("SELECT * FROM claims_fts LIMIT 1")
                    except sqlite3.OperationalError:
                        use_fts = False

                if use_fts:
                    # Sanitize query for FTS5
                    safe_query = _sanitize_fts_query(query)
                    if not safe_query:
                        # Empty query after sanitization - return empty
                        return []

                    sql = """
                        SELECT c.*, e.*, rank FROM claims_fts f
                        JOIN claim_records c ON c.claim_id = f.claim_id
                        JOIN evidence_records e ON e.evidence_id = c.evidence_id
                        WHERE claims_fts MATCH ?
                    """
                    params = [safe_query]
                else:
                    # Fallback to LIKE search
                    sql = """
                        SELECT c.*, e.*, 0.0 as rank FROM claim_records c
                        JOIN evidence_records e ON e.evidence_id = c.evidence_id
                        WHERE c.claim_text LIKE ?
                    """
                    params = [f"%{query}%"]

                # Add filters
                if case_id:
                    sql += " AND c.case_id = ?"
                    params.append(case_id)
                if judge_id:
                    sql += " AND c.judge_id = ?"
                    params.append(judge_id)
                if person_id:
                    sql += " AND c.person_id = ?"
                    params.append(person_id)
                if entity_id:
                    sql += " AND c.entity_id = ?"
                    params.append(entity_id)
                if status:
                    sql += " AND c.status = ?"
                    params.append(status)

                if use_fts:
                    sql += " ORDER BY rank LIMIT ?"
                else:
                    sql += " ORDER BY c.created_at DESC LIMIT ?"
                params.append(safe_limit)

                rows = conn.execute(sql, params).fetchall()
                results = []
                for row in rows:
                    # Last column is rank/score
                    score_col = row[-1]
                    if use_fts:
                        # BM25 rank (lower is better) -> normalized score
                        score = max(0.0, min(1.0, 1.0 / (1.0 + abs(score_col))))
                    else:
                        score = 0.5  # Default score for LIKE search

                    # Split row into claim and evidence parts
                    # claim_records has 14 columns, evidence_records has 12 columns
                    claim = self._row_to_claim(row[:14])
                    evidence = self._row_to_evidence(row[14:26])
                    results.append((claim, evidence, score))

                return results

        except sqlite3.Error as e:
            logger.error(f"Failed to search claims with evidence: {e}")
            raise StorageError(f"Failed to search claims with evidence: {e}")

    def store_claim(self, record: ClaimRecord) -> ClaimRecord:
        """Store claim record. FTS index is maintained automatically by triggers."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Enable foreign key enforcement for this connection
                conn.execute("PRAGMA foreign_keys=ON")

                # Check if claim exists
                existing = conn.execute(
                    "SELECT evidence_id FROM claim_records WHERE claim_id = ?",
                    (record.claim_id,)
                ).fetchone()

                if existing:
                    # Check if evidence_id is changing
                    old_evidence_id = existing[0]
                    if old_evidence_id != record.evidence_id:
                        # Verify new evidence exists
                        evidence_exists = conn.execute(
                            "SELECT 1 FROM evidence_records WHERE evidence_id = ?",
                            (record.evidence_id,)
                        ).fetchone()
                        if not evidence_exists:
                            eid = record.evidence_id
                            raise EvidenceNotFoundError(
                                f"Cannot update claim: evidence {eid} not found"
                            )
                    # Update existing (preserves rowid, triggers handle FTS)
                    conn.execute(
                        """
                        UPDATE claim_records SET
                            evidence_id = ?, claim_text = ?, claim_type = ?,
                            case_id = ?, judge_id = ?, person_id = ?, entity_id = ?,
                            confidence = ?, status = ?, tags = ?, metadata = ?,
                            updated_at = ?
                        WHERE claim_id = ?
                        """,
                        (
                            record.evidence_id,
                            record.claim_text,
                            record.claim_type,
                            record.case_id,
                            record.judge_id,
                            record.person_id,
                            record.entity_id,
                            record.confidence,
                            record.status,
                            json.dumps(record.tags),
                            json.dumps(record.metadata),
                            record.updated_at.isoformat(),
                            record.claim_id,
                        ),
                    )
                else:
                    # Verify evidence exists before creating claim
                    evidence_exists = conn.execute(
                        "SELECT 1 FROM evidence_records WHERE evidence_id = ?",
                        (record.evidence_id,)
                    ).fetchone()
                    if not evidence_exists:
                        eid = record.evidence_id
                        raise EvidenceNotFoundError(
                            f"Cannot create claim: evidence {eid} not found"
                        )
                    # Insert new (triggers handle FTS)
                    conn.execute(
                        """
                        INSERT INTO claim_records (
                            claim_id, evidence_id, claim_text, claim_type,
                            case_id, judge_id, person_id, entity_id,
                            confidence, status, tags, metadata, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            record.claim_id,
                            record.evidence_id,
                            record.claim_text,
                            record.claim_type,
                            record.case_id,
                            record.judge_id,
                            record.person_id,
                            record.entity_id,
                            record.confidence,
                            record.status,
                            json.dumps(record.tags),
                            json.dumps(record.metadata),
                            record.created_at.isoformat(),
                            record.updated_at.isoformat(),
                        ),
                    )
                conn.commit()
                return record
        except sqlite3.Error as e:
            logger.error("Failed to store claim: %s", e)
            raise StorageError(f"Failed to store claim: {e}")
    
    def get_claim(self, claim_id: str) -> Optional[ClaimRecord]:
        """Retrieve claim by ID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT * FROM claim_records WHERE claim_id = ?",
                    (claim_id,),
                ).fetchone()
                
                if not row:
                    return None
                    
                return self._row_to_claim(row)
        except sqlite3.Error as e:
            logger.error(f"Failed to get claim: {e}")
            raise StorageError(f"Failed to get claim: {e}")
    
    def _row_to_evidence(self, row: tuple) -> EvidenceRecord:
        """Convert database row to EvidenceRecord."""
        return EvidenceRecord(
            evidence_id=row[0],
            content_hash=row[1],
            source_type=row[2],
            source_url=row[3],
            source_title=row[4],
            content_preview=row[5],
            jurisdiction=row[6],
            published_at=datetime.fromisoformat(row[7]) if row[7] else None,
            file_path=row[8],
            metadata=json.loads(row[9]) if row[9] else {},
            ingested_at=datetime.fromisoformat(row[10]),
        )
    
    def _clamp_limit(self, limit: int, default: int = 50, maximum: int = 500) -> int:
        try:
            value = int(limit)
        except (TypeError, ValueError):
            return default
        return max(1, min(value, maximum))

    def _row_to_claim(self, row: tuple) -> ClaimRecord:
        """Convert database row to ClaimRecord."""
        return ClaimRecord(
            claim_id=row[0],
            evidence_id=row[1],
            claim_text=row[2],
            claim_type=row[3],
            case_id=row[4],
            judge_id=row[5],
            person_id=row[6],
            entity_id=row[7],
            confidence=row[8],
            status=row[9],
            tags=json.loads(row[10]) if row[10] else [],
            metadata=json.loads(row[11]) if row[11] else {},
            created_at=datetime.fromisoformat(row[12]),
            updated_at=datetime.fromisoformat(row[13]),
        )

    def search_claims(
        self,
        query: Optional[str] = None,
        case_id: Optional[str] = None,
        judge_id: Optional[str] = None,
        person_id: Optional[str] = None,
        entity_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> List[ClaimRecord]:
        """Search claim records."""
        try:
            sql = "SELECT * FROM claim_records WHERE 1=1"
            params = []

            if query:
                sql += " AND claim_text LIKE ?"
                params.append(f"%{query}%")

            if case_id:
                sql += " AND case_id = ?"
                params.append(case_id)

            if judge_id:
                sql += " AND judge_id = ?"
                params.append(judge_id)

            if person_id:
                sql += " AND person_id = ?"
                params.append(person_id)

            if entity_id:
                sql += " AND entity_id = ?"
                params.append(entity_id)

            if status:
                sql += " AND status = ?"
                params.append(status)

            safe_limit = self._clamp_limit(limit)
            sql += " ORDER BY created_at DESC LIMIT ?"
            params.append(safe_limit)

            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(sql, params).fetchall()
                return [self._row_to_claim(row) for row in rows]

        except sqlite3.Error as e:
            logger.error(f"Failed to search claims: {e}")
            raise StorageError(f"Failed to search claims: {e}")

    def get_claims_by_evidence(
        self,
        evidence_id: str,
        limit: int = 100,
    ) -> List[ClaimRecord]:
        """Get all claims linked to one evidence record."""
        try:
            safe_limit = self._clamp_limit(limit, default=100, maximum=500)
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    """
                    SELECT * FROM claim_records
                    WHERE evidence_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (evidence_id, safe_limit),
                ).fetchall()
                return [self._row_to_claim(row) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Failed to get claims by evidence: {e}")
            raise StorageError(f"Failed to get claims by evidence: {e}")

    def get_timeline_events(
        self,
        case_id: Optional[str] = None,
        judge_id: Optional[str] = None,
        person_id: Optional[str] = None,
        entity_id: Optional[str] = None,
        jurisdiction: Optional[str] = None,
        limit: int = 50,
    ) -> List[TimelineEvent]:
        """Get timeline events for entity/case/judge.

        Returns chronological events filtered by provided criteria.
        """
        try:
            sql = "SELECT * FROM timeline_events WHERE 1=1"
            params = []

            if case_id:
                sql += " AND case_id = ?"
                params.append(case_id)

            if judge_id:
                sql += " AND judge_id = ?"
                params.append(judge_id)

            if person_id:
                sql += " AND person_id = ?"
                params.append(person_id)

            if entity_id:
                sql += " AND entity_id = ?"
                params.append(entity_id)

            if jurisdiction:
                sql += " AND jurisdiction = ?"
                params.append(jurisdiction)

            safe_limit = self._clamp_limit(limit)
            sql += " ORDER BY event_date ASC LIMIT ?"
            params.append(safe_limit)

            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(sql, params).fetchall()
                return [self._row_to_timeline_event(row) for row in rows]

        except sqlite3.Error as e:
            logger.error(f"Failed to get timeline events: {e}")
            raise StorageError(f"Failed to get timeline events: {e}")

    def healthcheck(self) -> dict:
        """Return storage health."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("SELECT 1").fetchone()
            return {
                "status": "ok",
                "db_path": self.db_path,
                "evidence_dir": str(self.config.evidence_dir),
            }
        except sqlite3.Error as e:
            return {
                "status": "error",
                "db_path": self.db_path,
                "error": str(e),
            }

    def _row_to_timeline_event(self, row: tuple) -> TimelineEvent:
        """Convert database row to TimelineEvent."""
        return TimelineEvent(
            event_id=row[0],
            event_type=row[1],
            event_date=datetime.fromisoformat(row[2]),
            description=row[3],
            evidence_id=row[4],
            claim_id=row[5],
            case_id=row[6],
            judge_id=row[7],
            person_id=row[8],
            entity_id=row[9],
            jurisdiction=row[10],
            metadata=json.loads(row[11]) if row[11] else {},
        )
