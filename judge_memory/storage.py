"""Judge Memory SQLite Storage"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from judge_memory.config import JudgeMemoryConfig
from judge_memory.models import EvidenceRecord, ClaimRecord, TimelineEvent
from judge_memory.exceptions import StorageError
from judge_memory._logger import get_logger

logger = get_logger(__name__)


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
            
            conn.commit()
            logger.info(f"Database initialized: {self.db_path}")
    
    def store_evidence(self, record: EvidenceRecord) -> EvidenceRecord:
        """Store evidence record."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO evidence_records (
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
            raise StorageError(f"Failed to store evidence: {e}") from e
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
            
            sql += f" ORDER BY ingested_at DESC LIMIT {limit}"
            
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(sql, params).fetchall()
                return [self._row_to_evidence(row) for row in rows]
                
        except sqlite3.Error as e:
            logger.error(f"Failed to search evidence: {e}")
            raise StorageError(f"Failed to search evidence: {e}")
    
    def store_claim(self, record: ClaimRecord) -> ClaimRecord:
        """Store claim record."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO claim_records (
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
            logger.error(f"Failed to store claim: {e}")
            raise StorageError(f"Failed to store claim: {e}") from e
            logger.error(f"Failed to store claim: {e}")
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

            sql += f" ORDER BY created_at DESC LIMIT {limit}"

            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(sql, params).fetchall()
                return [self._row_to_claim(row) for row in rows]

        except sqlite3.Error as e:
            logger.error(f"Failed to search claims: {e}")
            raise StorageError(f"Failed to search claims: {e}")

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

            sql += f" ORDER BY event_date ASC LIMIT {limit}"

            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(sql, params).fetchall()
                return [self._row_to_timeline_event(row) for row in rows]

        except sqlite3.Error as e:
            logger.error(f"Failed to get timeline events: {e}")
            raise StorageError(f"Failed to get timeline events: {e}")

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
