"""Fluid Memory SQLite Storage"""

import json
import sqlite3
import hashlib
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from fluid_memory.models import MemoryItem
from fluid_memory.config import FluidMemoryConfig
from fluid_memory.exceptions import StorageError


def compute_sha256(content: str) -> str:
    """Compute SHA256 hash of content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class FluidMemoryStorage:
    """SQLite storage for fluid memory items."""

    def __init__(self, config: FluidMemoryConfig):
        self.config = config
        self._init_database()

    def _init_database(self) -> None:
        """Initialize SQLite database."""
        self.config.data_dir.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(str(self.config.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_items (
                    memory_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    content_hash TEXT UNIQUE NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_accessed_at TEXT NOT NULL,
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
            conn.commit()

    def _row_to_memory(self, row: tuple) -> MemoryItem:
        """Convert database row to MemoryItem."""
        return MemoryItem(
            memory_id=row[0],
            content=row[1],
            content_hash=row[2],
            created_at=datetime.fromisoformat(row[3]),
            updated_at=datetime.fromisoformat(row[4]),
            last_accessed_at=datetime.fromisoformat(row[5]),
            access_count=row[6],
            salience=row[7],
            confidence=row[8],
            volatility=row[9],
            stability=row[10],
            decay_rate=row[11],
            legal_salience=row[12],
            trust_salience=row[13],
            interest_salience=row[14],
            attention_salience=row[15],
            reinforcement_count=row[16],
            contradiction_count=row[17],
            source_refs=json.loads(row[18]) if row[18] else [],
            tags=json.loads(row[19]) if row[19] else [],
            links=json.loads(row[20]) if row[20] else [],
            metadata=json.loads(row[21]) if row[21] else {},
        )

    def store(self, memory: MemoryItem) -> MemoryItem:
        """Store or update a memory item."""
        try:
            with sqlite3.connect(str(self.config.db_path)) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO memory_items (
                        memory_id, content, content_hash, created_at, updated_at,
                        last_accessed_at, access_count, salience, confidence,
                        volatility, stability, decay_rate, legal_salience,
                        trust_salience, interest_salience, attention_salience,
                        reinforcement_count, contradiction_count, source_refs,
                        tags, links, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        memory.memory_id,
                        memory.content,
                        memory.content_hash,
                        memory.created_at.isoformat(),
                        memory.updated_at.isoformat(),
                        memory.last_accessed_at.isoformat(),
                        memory.access_count,
                        memory.salience,
                        memory.confidence,
                        memory.volatility,
                        memory.stability,
                        memory.decay_rate,
                        memory.legal_salience,
                        memory.trust_salience,
                        memory.interest_salience,
                        memory.attention_salience,
                        memory.reinforcement_count,
                        memory.contradiction_count,
                        json.dumps(memory.source_refs),
                        json.dumps(memory.tags),
                        json.dumps(memory.links),
                        json.dumps(memory.metadata),
                    ),
                )
                conn.commit()
                return memory
        except sqlite3.Error as e:
            raise StorageError(f"Failed to store memory: {e}")

    def get_by_hash(self, content_hash: str) -> Optional[MemoryItem]:
        """Get memory by content hash (for deduplication)."""
        try:
            with sqlite3.connect(str(self.config.db_path)) as conn:
                row = conn.execute(
                    "SELECT * FROM memory_items WHERE content_hash = ?",
                    (content_hash,),
                ).fetchone()
                return self._row_to_memory(row) if row else None
        except sqlite3.Error as e:
            raise StorageError(f"Failed to get memory by hash: {e}")

    def get_by_id(self, memory_id: str) -> Optional[MemoryItem]:
        """Get memory by ID."""
        try:
            with sqlite3.connect(str(self.config.db_path)) as conn:
                row = conn.execute(
                    "SELECT * FROM memory_items WHERE memory_id = ?",
                    (memory_id,),
                ).fetchone()
                return self._row_to_memory(row) if row else None
        except sqlite3.Error as e:
            raise StorageError(f"Failed to get memory: {e}")

    def search(
        self,
        query: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[MemoryItem]:
        """Search memories by keyword or tags."""
        try:
            sql = "SELECT * FROM memory_items WHERE 1=1"
            params = []

            if query:
                sql += " AND content LIKE ?"
                params.append(f"%{query}%")

            if tags:
                # Simple tag search - exact match in JSON array
                for tag in tags:
                    sql += " AND tags LIKE ?"
                    params.append(f'%"{tag}"%')

            sql += f" ORDER BY salience DESC LIMIT {limit}"

            with sqlite3.connect(str(self.config.db_path)) as conn:
                rows = conn.execute(sql, params).fetchall()
                return [self._row_to_memory(row) for row in rows]
        except sqlite3.Error as e:
            raise StorageError(f"Failed to search memories: {e}")

    def get_all(self, limit: int = 100) -> List[MemoryItem]:
        """Get all memories."""
        try:
            with sqlite3.connect(str(self.config.db_path)) as conn:
                rows = conn.execute(
                    f"SELECT * FROM memory_items ORDER BY updated_at DESC LIMIT {limit}"
                ).fetchall()
                return [self._row_to_memory(row) for row in rows]
        except sqlite3.Error as e:
            raise StorageError(f"Failed to get memories: {e}")
