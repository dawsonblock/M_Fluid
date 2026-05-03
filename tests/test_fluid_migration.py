"""
Test Fluid Memory schema migration from old versions.

Verifies that databases created without the new checksum/invalidation
columns are automatically migrated when opened with the new code.
"""

import tempfile
import sqlite3
import pytest
from pathlib import Path

from fluid_memory.storage import MemoryStorage
from fluid_memory.models import MemoryItem


def create_old_schema_db(db_path: str) -> None:
    """Create a database with the old schema (no checksum/invalidation columns)."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        # Old schema without state_checksum, invalidated_at, invalidation_reason
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_items (
                memory_id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                content_hash TEXT UNIQUE NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                last_accessed_at REAL,
                access_count INTEGER DEFAULT 0,
                salience REAL DEFAULT 0.5,
                confidence REAL DEFAULT 0.5,
                volatility REAL DEFAULT 0.3,
                stability REAL DEFAULT 0.5,
                decay_rate REAL DEFAULT 0.05,
                legal_salience REAL DEFAULT 0.5,
                trust_salience REAL DEFAULT 0.5,
                interest_salience REAL DEFAULT 0.5,
                attention_salience REAL DEFAULT 0.5,
                reinforcement_count INTEGER DEFAULT 0,
                contradiction_count INTEGER DEFAULT 0,
                source_refs TEXT,
                tags TEXT,
                links TEXT,
                metadata TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_events (
                event_id TEXT PRIMARY KEY,
                memory_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                timestamp REAL NOT NULL,
                delta_json TEXT,
                metadata_json TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_links (
                link_id TEXT PRIMARY KEY,
                source_memory_id TEXT NOT NULL,
                target_memory_id TEXT NOT NULL,
                link_type TEXT NOT NULL,
                strength REAL DEFAULT 0.5
            )
        """)
        conn.commit()


def test_migration_adds_checksum_column():
    """Test that opening an old database adds the state_checksum column."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        create_old_schema_db(str(db_path))

        # Verify old schema doesn't have checksum column
        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.execute("PRAGMA table_info(memory_items)")
            columns = {row[1] for row in cursor.fetchall()}
            assert "state_checksum" not in columns
            assert "invalidated_at" not in columns
            assert "invalidation_reason" not in columns

        # Open with new storage - should migrate
        storage = MemoryStorage(str(db_path))

        # Verify new columns exist
        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.execute("PRAGMA table_info(memory_items)")
            columns = {row[1] for row in cursor.fetchall()}
            assert "state_checksum" in columns
            assert "invalidated_at" in columns
            assert "invalidation_reason" in columns


def test_migration_preserves_existing_data():
    """Test that migration preserves existing memory data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        create_old_schema_db(str(db_path))

        # Insert some data with old schema
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("""
                INSERT INTO memory_items (
                    memory_id, content, content_hash, created_at, updated_at, salience
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, ("mem1", "Test content", "abc123", 1000.0, 1000.0, 0.7))
            conn.commit()

        # Open with new storage - should migrate and preserve data
        storage = MemoryStorage(str(db_path))

        # Verify data is preserved
        memory = storage.get_memory("mem1")
        assert memory is not None
        assert memory.content == "Test content"
        assert memory.content_hash == "abc123"
        assert memory.salience == 0.7


def test_checksum_methods_work_after_migration():
    """Test that checksum operations work after migration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        create_old_schema_db(str(db_path))

        # Insert data
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("""
                INSERT INTO memory_items (
                    memory_id, content, content_hash, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?)
            """, ("mem1", "Test content", "abc123", 1000.0, 1000.0))
            conn.commit()

        # Open with new storage
        storage = MemoryStorage(str(db_path))

        # Should be able to update checksum
        checksum = storage.update_checksum("mem1")
        assert checksum is not None
        assert len(checksum) == 16

        # Should be able to verify checksum
        assert storage.verify_checksum("mem1") is True


def test_invalidation_methods_work_after_migration():
    """Test that invalidation operations work after migration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        create_old_schema_db(str(db_path))

        # Insert data
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("""
                INSERT INTO memory_items (
                    memory_id, content, content_hash, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?)
            """, ("mem1", "Test content", "abc123", 1000.0, 1000.0))
            conn.commit()

        # Open with new storage
        storage = MemoryStorage(str(db_path))

        # Should be able to invalidate
        result = storage.invalidate("mem1", reason="Test invalidation")
        assert result is True

        # Should be able to check invalidation
        assert storage.is_invalidated("mem1") is True

        # Should be in invalidated list
        invalidated = storage.get_invalidated_memories()
        assert len(invalidated) == 1
        assert invalidated[0].memory_id == "mem1"


def test_new_database_has_all_columns():
    """Test that newly created databases have all columns from the start."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"

        # Create new storage (should create fresh schema)
        storage = MemoryStorage(str(db_path))

        # Add a memory
        memory = MemoryItem(content="New memory")
        storage.save_memory(memory)

        # Verify all columns exist
        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.execute("PRAGMA table_info(memory_items)")
            columns = {row[1] for row in cursor.fetchall()}
            assert "state_checksum" in columns
            assert "invalidated_at" in columns
            assert "invalidation_reason" in columns

        # Verify checksum/invalidation work
        checksum = storage.update_checksum(memory.memory_id)
        assert checksum is not None

        result = storage.invalidate(memory.memory_id, reason="Test")
        assert result is True
