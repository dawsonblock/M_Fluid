"""Fluid Memory SQLite Storage"""

import json
import sqlite3
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any

from fluid_memory.models import MemoryItem, MemoryLink
from fluid_memory.events import MemoryEvent, EventType
from fluid_memory.config import FluidMemoryConfig
from fluid_memory.exceptions import StorageError


def compute_sha256(content: str) -> str:
    """Compute SHA256 hash of content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# MemoryStorage – simple, dependency-light interface used by tests and engine
# ---------------------------------------------------------------------------

class MemoryStorage:
    """SQLite-backed storage with a straightforward CRUD + event + link API.

    Args:
        db_path: Filesystem path to the SQLite database file (string or Path).
    """

    def __init__(self, db_path: str):
        self.db_path = str(db_path)
        self._init_database()

    def _init_database(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
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

    # ------------------------------------------------------------------
    # Memory CRUD
    # ------------------------------------------------------------------

    def _row_to_memory(self, row: tuple) -> MemoryItem:
        return MemoryItem(
            memory_id=row[0],
            content=row[1],
            content_hash=row[2],
            created_at=row[3],
            updated_at=row[4],
            last_accessed_at=row[5],
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

    def _memory_to_params(self, memory: MemoryItem) -> tuple:
        return (
            memory.memory_id,
            memory.content,
            memory.content_hash,
            memory.created_at,
            memory.updated_at,
            memory.last_accessed_at,
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
        )

    def save_memory(self, memory: MemoryItem) -> MemoryItem:
        """Insert a new memory item."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """INSERT INTO memory_items (
                        memory_id, content, content_hash, created_at, updated_at,
                        last_accessed_at, access_count, salience, confidence,
                        volatility, stability, decay_rate, legal_salience,
                        trust_salience, interest_salience, attention_salience,
                        reinforcement_count, contradiction_count, source_refs,
                        tags, links, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    self._memory_to_params(memory),
                )
                conn.commit()
            return memory
        except sqlite3.Error as e:
            raise StorageError(f"Failed to save memory: {e}")

    def get_memory(self, memory_id: str) -> Optional[MemoryItem]:
        """Retrieve a memory by ID, or None if not found."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT * FROM memory_items WHERE memory_id = ?", (memory_id,)
                ).fetchone()
            return self._row_to_memory(row) if row else None
        except sqlite3.Error as e:
            raise StorageError(f"Failed to get memory: {e}")

    def get_memory_by_hash(self, content_hash: str) -> Optional[MemoryItem]:
        """Retrieve a memory by content hash, or None if not found."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT * FROM memory_items WHERE content_hash = ?", (content_hash,)
                ).fetchone()
            return self._row_to_memory(row) if row else None
        except sqlite3.Error as e:
            raise StorageError(f"Failed to get memory by hash: {e}")

    def update_memory(self, memory: MemoryItem) -> MemoryItem:
        """Update an existing memory item."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """UPDATE memory_items SET
                        content=?, content_hash=?, created_at=?, updated_at=?,
                        last_accessed_at=?, access_count=?, salience=?, confidence=?,
                        volatility=?, stability=?, decay_rate=?, legal_salience=?,
                        trust_salience=?, interest_salience=?, attention_salience=?,
                        reinforcement_count=?, contradiction_count=?, source_refs=?,
                        tags=?, links=?, metadata=?
                    WHERE memory_id=?""",
                    (
                        memory.content,
                        memory.content_hash,
                        memory.created_at,
                        memory.updated_at,
                        memory.last_accessed_at,
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
                        memory.memory_id,
                    ),
                )
                conn.commit()
            return memory
        except sqlite3.Error as e:
            raise StorageError(f"Failed to update memory: {e}")

    def delete_memory(self, memory_id: str) -> None:
        """Delete a memory item by ID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM memory_items WHERE memory_id = ?", (memory_id,))
                conn.commit()
        except sqlite3.Error as e:
            raise StorageError(f"Failed to delete memory: {e}")

    def search_memories(
        self,
        query: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[MemoryItem]:
        """Keyword + tag search."""
        try:
            sql = "SELECT * FROM memory_items WHERE 1=1"
            params: list = []

            if query:
                sql += " AND content LIKE ?"
                params.append(f"%{query}%")

            if tags:
                for tag in tags:
                    sql += ' AND tags LIKE ?'
                    params.append(f'%"{tag}"%')

            sql += " ORDER BY salience DESC LIMIT ?"
            params.append(int(limit))

            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(sql, params).fetchall()
            return [self._row_to_memory(row) for row in rows]
        except sqlite3.Error as e:
            raise StorageError(f"Failed to search memories: {e}")

    def get_all(self, limit: int = 100) -> List[MemoryItem]:
        """Return all memories up to limit."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT * FROM memory_items ORDER BY updated_at DESC LIMIT ?",
                    (int(limit),),
                ).fetchall()
            return [self._row_to_memory(row) for row in rows]
        except sqlite3.Error as e:
            raise StorageError(f"Failed to get memories: {e}")

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def save_event(self, event: MemoryEvent) -> None:
        """Append an event record."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """INSERT INTO memory_events
                        (event_id, memory_id, event_type, timestamp, delta_json, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        event.event_id,
                        event.memory_id,
                        event.event_type.value,
                        event.timestamp,
                        json.dumps(event.delta_json),
                        json.dumps(event.metadata_json),
                    ),
                )
                conn.commit()
        except sqlite3.Error as e:
            raise StorageError(f"Failed to save event: {e}")

    def get_events(self, memory_id: str) -> List[MemoryEvent]:
        """Retrieve all events for a memory, ordered by timestamp."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT * FROM memory_events WHERE memory_id = ? ORDER BY timestamp",
                    (memory_id,),
                ).fetchall()
            return [
                MemoryEvent(
                    event_id=row[0],
                    memory_id=row[1],
                    event_type=EventType(row[2]),
                    timestamp=row[3],
                    delta_json=json.loads(row[4]) if row[4] else {},
                    metadata_json=json.loads(row[5]) if row[5] else {},
                )
                for row in rows
            ]
        except sqlite3.Error as e:
            raise StorageError(f"Failed to get events: {e}")

    # ------------------------------------------------------------------
    # Links
    # ------------------------------------------------------------------

    def save_link(self, link: MemoryLink) -> None:
        """Store a memory link."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO memory_links
                        (link_id, source_memory_id, target_memory_id, link_type, strength)
                    VALUES (?, ?, ?, ?, ?)""",
                    (
                        link.link_id,
                        link.source_memory_id,
                        link.target_memory_id,
                        link.link_type,
                        link.strength,
                    ),
                )
                conn.commit()
        except sqlite3.Error as e:
            raise StorageError(f"Failed to save link: {e}")

    def get_links(self, memory_id: str) -> List[MemoryLink]:
        """Retrieve links originating from a memory."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT * FROM memory_links WHERE source_memory_id = ?",
                    (memory_id,),
                ).fetchall()
            return [
                MemoryLink(
                    link_id=row[0],
                    source_memory_id=row[1],
                    target_memory_id=row[2],
                    link_type=row[3],
                    strength=row[4],
                )
                for row in rows
            ]
        except sqlite3.Error as e:
            raise StorageError(f"Failed to get links: {e}")


# ---------------------------------------------------------------------------
# FluidMemoryStorage – config-based wrapper kept for backward compatibility
# ---------------------------------------------------------------------------

class FluidMemoryStorage(MemoryStorage):
    """Config-based storage wrapper (backward-compatible with FluidMemoryEngine)."""

    def __init__(self, config: FluidMemoryConfig):
        self.config = config
        super().__init__(str(config.db_path))

    # Alias names used by the legacy engine
    def store(self, memory: MemoryItem) -> MemoryItem:
        """Insert or replace a memory item."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO memory_items (
                        memory_id, content, content_hash, created_at, updated_at,
                        last_accessed_at, access_count, salience, confidence,
                        volatility, stability, decay_rate, legal_salience,
                        trust_salience, interest_salience, attention_salience,
                        reinforcement_count, contradiction_count, source_refs,
                        tags, links, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    self._memory_to_params(memory),
                )
                conn.commit()
            return memory
        except sqlite3.Error as e:
            raise StorageError(f"Failed to store memory: {e}")

    def get_by_hash(self, content_hash: str) -> Optional[MemoryItem]:
        return self.get_memory_by_hash(content_hash)

    def get_by_id(self, memory_id: str) -> Optional[MemoryItem]:
        return self.get_memory(memory_id)

    def search(
        self,
        query: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[MemoryItem]:
        return self.search_memories(query=query, tags=tags, limit=limit)
