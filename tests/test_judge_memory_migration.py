"""
Test Judge Memory schema migration and FTS backfill.

Verifies that:
1. FTS tables are created automatically
2. Existing data is backfilled into FTS indexes
3. FTS triggers work for new inserts after migration
"""

import tempfile
import sqlite3
from datetime import datetime
from pathlib import Path

from judge_memory.storage import JudgeMemoryStorage
from judge_memory.models import EvidenceRecord, ClaimRecord


def create_db_without_fts(db_path: str) -> None:
    """Create a database without FTS tables (simulating old version)."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        # Evidence table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS evidence_records (
                evidence_id TEXT PRIMARY KEY,
                content_hash TEXT UNIQUE NOT NULL,
                source_type TEXT,
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
                updated_at TEXT NOT NULL
            )
        """)
        # Timeline table
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
        conn.commit()


def test_fts_tables_created_on_migration():
    """Test that FTS tables are created when opening old database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "judge.db"
        create_db_without_fts(str(db_path))

        # Verify no FTS tables exist
        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {row[0] for row in cursor.fetchall()}
            assert "evidence_fts" not in tables
            assert "claims_fts" not in tables

        # Open with new storage - should create FTS tables
        from judge_memory.config import JudgeMemoryConfig
        config = JudgeMemoryConfig(sqlite_path=str(db_path))
        storage = JudgeMemoryStorage(config)

        # Verify FTS tables now exist
        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {row[0] for row in cursor.fetchall()}
            assert "evidence_fts" in tables
            assert "claims_fts" in tables


def test_existing_data_backfilled():
    """Test that existing evidence/claims are backfilled into FTS."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "judge.db"
        create_db_without_fts(str(db_path))

        # Insert data without FTS
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("""
                INSERT INTO evidence_records (
                    evidence_id, content_hash, source_title, content_preview,
                    source_url, ingested_at
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, ("ev1", "hash1", "Court Decision", "This is about bail proceedings",
                  "http://example.com", datetime.now().isoformat()))

            conn.execute("""
                INSERT INTO claim_records (
                    claim_id, evidence_id, claim_text, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?)
            """, ("cl1", "ev1", "Saskatoon court granted bail", datetime.now().isoformat(),
                  datetime.now().isoformat()))
            conn.commit()

        # Open with new storage - should backfill
        from judge_memory.config import JudgeMemoryConfig
        config = JudgeMemoryConfig(sqlite_path=str(db_path))
        storage = JudgeMemoryStorage(config)

        # Verify FTS indexes have the backfilled data
        with sqlite3.connect(str(db_path)) as conn:
            # Check evidence FTS
            cursor = conn.execute("SELECT COUNT(*) FROM evidence_fts")
            assert cursor.fetchone()[0] == 1

            # Check claims FTS
            cursor = conn.execute("SELECT COUNT(*) FROM claims_fts")
            assert cursor.fetchone()[0] == 1

            # Verify content is searchable
            cursor = conn.execute(
                "SELECT * FROM evidence_fts WHERE evidence_fts MATCH ?",
                ("bail",)
            )
            assert cursor.fetchone() is not None

            cursor = conn.execute(
                "SELECT * FROM claims_fts WHERE claims_fts MATCH ?",
                ("Saskatoon",)
            )
            assert cursor.fetchone() is not None


def test_new_data_indexed_via_triggers():
    """Test that new inserts after migration are automatically indexed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "judge.db"
        create_db_without_fts(str(db_path))

        # Add some old data
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("""
                INSERT INTO evidence_records (
                    evidence_id, content_hash, source_title, content_preview,
                    source_url, ingested_at
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, ("ev1", "hash1", "Old Evidence", "Old content",
                  "http://old.com", datetime.now().isoformat()))
            conn.commit()

        # Open with new storage (triggers now exist)
        from judge_memory.config import JudgeMemoryConfig
        config = JudgeMemoryConfig(sqlite_path=str(db_path))
        storage = JudgeMemoryStorage(config)

        # Add new evidence via storage (should trigger FTS insert)
        new_ev = EvidenceRecord(
            evidence_id="ev2",
            content_hash="hash2",
            source_type="court_document",
            source_title="New Court Ruling",
            content_preview="This is new evidence about criminal proceedings",
            source_url="http://new.com",
            ingested_at=datetime.now(),
        )
        storage.store_evidence(new_ev)

        # Add new claim via storage (should trigger FTS insert)
        new_claim = ClaimRecord(
            claim_id="cl2",
            evidence_id="ev2",
            claim_text="Criminal charges were dismissed by the court",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        storage.store_claim(new_claim)

        # Verify both are searchable
        with sqlite3.connect(str(db_path)) as conn:
            # New evidence should be searchable
            cursor = conn.execute(
                "SELECT * FROM evidence_fts WHERE evidence_fts MATCH ?",
                ("criminal",)
            )
            assert cursor.fetchone() is not None

            # New claim should be searchable
            cursor = conn.execute(
                "SELECT * FROM claims_fts WHERE claims_fts MATCH ?",
                ("dismissed",)
            )
            assert cursor.fetchone() is not None

            # Count total
            cursor = conn.execute("SELECT COUNT(*) FROM evidence_fts")
            assert cursor.fetchone()[0] == 2  # Old + new

            cursor = conn.execute("SELECT COUNT(*) FROM claims_fts")
            assert cursor.fetchone()[0] == 1  # Just new


def test_claim_update_works_with_triggers():
    """Test that updating a claim updates the FTS index correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "judge.db"

        from judge_memory.config import JudgeMemoryConfig
        config = JudgeMemoryConfig(sqlite_path=str(db_path))
        storage = JudgeMemoryStorage(config)

        # Add a claim (need evidence first)
        ev = EvidenceRecord(
            evidence_id="ev1",
            content_hash="hash1",
            source_type="court",
            source_title="Court Doc",
            content_preview="Court proceedings document",
            ingested_at=datetime.now(),
        )
        storage.store_evidence(ev)

        claim = ClaimRecord(
            claim_id="cl1",
            evidence_id="ev1",
            claim_text="Original claim about court proceedings",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        storage.store_claim(claim)

        # Verify original text is searchable
        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.execute(
                "SELECT * FROM claims_fts WHERE claims_fts MATCH ?",
                ("Original",)
            )
            assert cursor.fetchone() is not None

        # Update the claim text
        claim.claim_text = "Updated claim about bail hearing"
        claim.updated_at = datetime.now()
        storage.store_claim(claim)

        # Verify updated text is searchable
        with sqlite3.connect(str(db_path)) as conn:
            # New text should be found
            cursor = conn.execute(
                "SELECT * FROM claims_fts WHERE claims_fts MATCH ?",
                ("bail",)
            )
            assert cursor.fetchone() is not None

            # Old text should NOT be found
            cursor = conn.execute(
                "SELECT * FROM claims_fts WHERE claims_fts MATCH ?",
                ("Original",)
            )
            assert cursor.fetchone() is None


def test_rebuild_fts_indexes_manual():
    """Test manual FTS rebuild command."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "judge.db"

        from judge_memory.config import JudgeMemoryConfig
        config = JudgeMemoryConfig(sqlite_path=str(db_path))
        storage = JudgeMemoryStorage(config)

        # Add evidence and claims
        ev = EvidenceRecord(
            evidence_id="ev1",
            content_hash="hash1",
            source_type="test",
            source_title="Test Evidence",
            content_preview="Test content",
            ingested_at=datetime.now(),
        )
        storage.store_evidence(ev)

        claim = ClaimRecord(
            claim_id="cl1",
            evidence_id="ev1",
            claim_text="Test claim",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        storage.store_claim(claim)

        # Rebuild
        result = storage.rebuild_fts_indexes()
        assert result["evidence_count"] == 1
        assert result["claims_count"] == 1

        # Verify data still searchable after rebuild
        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.execute(
                "SELECT * FROM evidence_fts WHERE evidence_fts MATCH ?",
                ("Test",)
            )
            assert cursor.fetchone() is not None
