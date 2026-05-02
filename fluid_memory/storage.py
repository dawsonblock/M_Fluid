"""
Fluid Memory SQLite Storage

SQLite-backed storage for memories, events, and links.
"""

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from fluid_memory.models import MemoryItem, MemoryLink
from fluid_memory.events import MemoryEvent, EventType
from fluid_memory.exceptions import StorageError, MemoryNotFoundError


class MemoryStorage:
    """
    SQLite storage backend for fluid memory.
    
    Manages three tables:
    - memories: Memory items with JSON-encoded list/dict fields
    - memory_events: Event log for all state changes
    - memory_links: Bidirectional links between memories
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize database tables."""
        with sqlite3.connect(self.db_path) as conn:
            # Memories table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    last_accessed_at REAL,
                    access_count INTEGER DEFAULT 0,
                    salience REAL DEFAULT 0.5,
                    confidence REAL DEFAULT 0.5,
                    volatility REAL DEFAULT 0.3,
                    stability REAL DEFAULT 0.5,
                    decay_rate REAL DEFAULT 0.05,
                    reinforcement_count INTEGER DEFAULT 0,
                    contradiction_count INTEGER DEFAULT 0,
                    source_refs TEXT DEFAULT '[]',
                    tags TEXT DEFAULT '[]',
                    links TEXT DEFAULT '[]',
                    metadata TEXT DEFAULT '{}'
                )
            """)
            
            # Events table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_events (
                    event_id TEXT PRIMARY KEY,
                    memory_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    delta_json TEXT DEFAULT '{}',
                    metadata_json TEXT DEFAULT '{}',
                    FOREIGN KEY (memory_id) REFERENCES memories(memory_id)
                )
            """)
            
            # Links table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_links (
                    link_id TEXT PRIMARY KEY,
                    source_memory_id TEXT NOT NULL,
                    target_memory_id TEXT NOT NULL,
                    link_type TEXT NOT NULL,
                    strength REAL DEFAULT 0.5,
                    metadata_json TEXT DEFAULT '{}',
                    created_at REAL NOT NULL
                )
            """)
            
            # Indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_hash ON memories(content_hash)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_salience ON memories(salience)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_memory ON memory_events(memory_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_links_source ON memory_links(source_memory_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_links_target ON memory_links(target_memory_id)")
            
            conn.commit()
    
    def _memory_to_row(self, item: MemoryItem) -> Tuple:
        """Convert MemoryItem to database row tuple."""
        return (
            item.memory_id,
            item.content,
            item.content_hash,
            item.created_at,
            item.updated_at,
            item.last_accessed_at,
            item.access_count,
            item.salience,
            item.confidence,
            item.volatility,
            item.stability,
            item.decay_rate,
            item.reinforcement_count,
            item.contradiction_count,
            json.dumps(item.source_refs),
            json.dumps(item.tags),
            json.dumps(item.links),
            json.dumps(item.metadata),
        )
    
    def _row_to_memory(self, row: Tuple) -> MemoryItem:
        """Convert database row to MemoryItem."""
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
            reinforcement_count=row[12],
            contradiction_count=row[13],
            source_refs=json.loads(row[14]),
            tags=json.loads(row[15]),
            links=json.loads(row[16]),
            metadata=json.loads(row[17]),
        )
    
    def save_memory(self, item: MemoryItem) -> None:
        """Save a new memory item."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """INSERT INTO memories VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    self._memory_to_row(item)
                )
                conn.commit()
        except sqlite3.IntegrityError as e:
            raise StorageError(f"Failed to save memory: {e}")
    
    def get_memory(self, memory_id: str) -> Optional[MemoryItem]:
        """Retrieve a memory by ID."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM memories WHERE memory_id = ?",
                (memory_id,)
            ).fetchone()
            if row:
                return self._row_to_memory(row)
            return None
    
    def get_memory_by_hash(self, content_hash: str) -> Optional[MemoryItem]:
        """Retrieve a memory by content hash."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM memories WHERE content_hash = ?",
                (content_hash,)
            ).fetchone()
            if row:
                return self._row_to_memory(row)
            return None
    
    def update_memory(self, item: MemoryItem) -> None:
        """Update an existing memory."""
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute(
                """UPDATE memories SET
                    content = ?, content_hash = ?, updated_at = ?, last_accessed_at = ?,
                    access_count = ?, salience = ?, confidence = ?, volatility = ?,
                    stability = ?, decay_rate = ?, reinforcement_count = ?,
                    contradiction_count = ?, source_refs = ?, tags = ?, links = ?, metadata = ?
                WHERE memory_id = ?""",
                (
                    item.content, item.content_hash, item.updated_at, item.last_accessed_at,
                    item.access_count, item.salience, item.confidence, item.volatility,
                    item.stability, item.decay_rate, item.reinforcement_count,
                    item.contradiction_count, json.dumps(item.source_refs),
                    json.dumps(item.tags), json.dumps(item.links), json.dumps(item.metadata),
                    item.memory_id
                )
            )
            if result.rowcount == 0:
                raise MemoryNotFoundError(f"Memory not found: {item.memory_id}")
            conn.commit()
    
    def delete_memory(self, memory_id: str) -> None:
        """Delete a memory and its events."""
        with sqlite3.connect(self.db_path) as conn:
            # Delete events first (foreign key)
            conn.execute("DELETE FROM memory_events WHERE memory_id = ?", (memory_id,))
            # Delete links
            conn.execute(
                "DELETE FROM memory_links WHERE source_memory_id = ? OR target_memory_id = ?",
                (memory_id, memory_id)
            )
            # Delete memory
            result = conn.execute("DELETE FROM memories WHERE memory_id = ?", (memory_id,))
            if result.rowcount == 0:
                raise MemoryNotFoundError(f"Memory not found: {memory_id}")
            conn.commit()
    
    def search_memories(
        self,
        query: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 20
    ) -> List[MemoryItem]:
        """Search memories by text content and/or tags."""
        with sqlite3.connect(self.db_path) as conn:
            if query and tags:
                # Text + tag search
                placeholders = ", ".join("?" * len(tags))
                pattern = f"%{query}%"
                rows = conn.execute(
                    f"""SELECT * FROM memories 
                    WHERE content LIKE ? 
                    AND (
                        SELECT COUNT(*) FROM json_each(tags) 
                        WHERE value IN ({placeholders})
                    ) > 0
                    ORDER BY salience DESC LIMIT ?""",
                    (pattern, *tags, limit)
                ).fetchall()
            elif query:
                # Text only
                pattern = f"%{query}%"
                rows = conn.execute(
                    "SELECT * FROM memories WHERE content LIKE ? ORDER BY salience DESC LIMIT ?",
                    (pattern, limit)
                ).fetchall()
            elif tags:
                # Tags only
                placeholders = ", ".join("?" * len(tags))
                rows = conn.execute(
                    f"""SELECT * FROM memories WHERE (
                        SELECT COUNT(*) FROM json_each(tags) 
                        WHERE value IN ({placeholders})
                    ) > 0 ORDER BY salience DESC LIMIT ?""",
                    (*tags, limit)
                ).fetchall()
            else:
                # No filters - return by salience
                rows = conn.execute(
                    "SELECT * FROM memories ORDER BY salience DESC LIMIT ?",
                    (limit,)
                ).fetchall()
            
            return [self._row_to_memory(row) for row in rows]
    
    def save_event(self, event: MemoryEvent) -> None:
        """Save a memory event."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO memory_events VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    event.event_id,
                    event.memory_id,
                    event.event_type.value,
                    event.timestamp,
                    json.dumps(event.delta_json),
                    json.dumps(event.metadata_json),
                )
            )
            conn.commit()
    
    def get_events(self, memory_id: str) -> List[MemoryEvent]:
        """Get all events for a memory."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM memory_events WHERE memory_id = ? ORDER BY timestamp",
                (memory_id,)
            ).fetchall()
            return [
                MemoryEvent(
                    event_id=row[0],
                    memory_id=row[1],
                    event_type=EventType(row[2]),
                    timestamp=row[3],
                    delta_json=json.loads(row[4]),
                    metadata_json=json.loads(row[5]),
                )
                for row in rows
            ]
    
    def save_link(self, link: MemoryLink) -> None:
        """Save a memory link."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO memory_links VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    link.link_id,
                    link.source_memory_id,
                    link.target_memory_id,
                    link.link_type,
                    link.strength,
                    json.dumps(link.metadata),
                    link.created_at,
                )
            )
            conn.commit()
    
    def get_links(self, memory_id: str) -> List[MemoryLink]:
        """Get all links for a memory (as source or target)."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """SELECT * FROM memory_links 
                WHERE source_memory_id = ? OR target_memory_id = ?""",
                (memory_id, memory_id)
            ).fetchall()
            return [
                MemoryLink(
                    link_id=row[0],
                    source_memory_id=row[1],
                    target_memory_id=row[2],
                    link_type=row[3],
                    strength=row[4],
                    metadata=json.loads(row[5]),
                    created_at=row[6],
                )
                for row in rows
            ]
    
    def get_all_memories(self) -> List[MemoryItem]:
        """Get all memories (for decay operations)."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT * FROM memories").fetchall()
            return [self._row_to_memory(row) for row in rows]
