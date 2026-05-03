"""Fluid Memory SQLite Storage"""

import json
import sqlite3
import hashlib
import pickle
import struct
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from time import time

from fluid_memory.models import MemoryItem, MemoryLink
from fluid_memory.events import MemoryEvent, EventType
from fluid_memory.config import FluidMemoryConfig
from fluid_memory.exceptions import StorageError


def compute_sha256(content: str) -> str:
    """Compute SHA256 hash of content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _compute_embedding(content: str, dim: int = 128) -> bytes:
    """Compute a simple content embedding using hash-based features.

    This is a lightweight embedding that doesn't require ML libraries.
    For production, replace with sentence-transformers or similar.

    Args:
        content: Text content to embed
        dim: Embedding dimension (default 128)

    Returns:
        Serialized embedding as bytes
    """
    # Simple feature extraction based on character n-grams
    content = content.lower()
    features = [0.0] * dim

    # Character bigram features
    for i in range(len(content) - 1):
        bigram = content[i:i+2]
        hash_val = hashlib.md5(bigram.encode()).digest()
        idx = struct.unpack('H', hash_val[:2])[0] % dim
        features[idx] += 1.0

    # Normalize
    norm = sum(f * f for f in features) ** 0.5
    if norm > 0:
        features = [f / norm for f in features]

    return pickle.dumps(features)


def _cosine_similarity(emb1: bytes, emb2: bytes) -> float:
    """Compute cosine similarity between two embeddings."""
    try:
        v1 = pickle.loads(emb1)
        v2 = pickle.loads(emb2)

        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = sum(a * a for a in v1) ** 0.5
        norm2 = sum(b * b for b in v2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot / (norm1 * norm2)
    except Exception:
        return 0.0


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
        self._migrate_database()

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
                    metadata TEXT,
                    state_checksum TEXT,
                    invalidated_at REAL,
                    invalidation_reason TEXT,
                    embedding BLOB
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_embeddings (
                    memory_id TEXT PRIMARY KEY,
                    embedding BLOB NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY (memory_id) REFERENCES memory_items(memory_id) ON DELETE CASCADE
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

    def _migrate_database(self) -> None:
        """Migrate existing database schema to add missing columns."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get existing columns
                cursor = conn.execute("PRAGMA table_info(memory_items)")
                existing_columns = {row[1] for row in cursor.fetchall()}

                # Columns to add if missing
                migrations = [
                    ("state_checksum", "TEXT"),
                    ("invalidated_at", "REAL"),
                    ("invalidation_reason", "TEXT"),
                    ("embedding", "BLOB"),
                ]

                for column_name, column_type in migrations:
                    if column_name not in existing_columns:
                        conn.execute(
                            f"ALTER TABLE memory_items ADD COLUMN {column_name} {column_type}"
                        )
                        conn.commit()
        except sqlite3.Error as e:
            # Log but don't fail - migrations are best-effort
            import logging

            logging.getLogger(__name__).warning(f"Database migration skipped: {e}")

    # ------------------------------------------------------------------
    # Memory CRUD
    # ------------------------------------------------------------------

    def _row_to_memory(self, row: tuple) -> MemoryItem:
        memory = MemoryItem(
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
        # Set explicit invalidation/checksum fields (not in constructor to avoid validation issues)
        if len(row) > 22 and row[22]:
            object.__setattr__(memory, "state_checksum", row[22])
        if len(row) > 23 and row[23]:
            object.__setattr__(memory, "invalidated_at", row[23])
        if len(row) > 24 and row[24]:
            object.__setattr__(memory, "invalidation_reason", row[24])
        return memory

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
            # Invalidation and checksum fields
            memory.state_checksum,
            memory.invalidated_at,
            memory.invalidation_reason,
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
                        tags, links, metadata, state_checksum, invalidated_at,
                        invalidation_reason
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    self._memory_to_params(memory),
                )
                conn.commit()
            return memory
        except sqlite3.Error as e:
            raise StorageError(f"Failed to save memory: {e}")

    def get_memory(
        self,
        memory_id: str,
        include_invalidated: bool = False,
    ) -> Optional[MemoryItem]:
        """Retrieve a memory by ID, or None if not found.

        Args:
            memory_id: Memory ID to retrieve
            include_invalidated: If False, returns None for invalidated memories

        Returns:
            MemoryItem if found, None otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                if include_invalidated:
                    # Return memory regardless of invalidation status
                    row = conn.execute(
                        "SELECT * FROM memory_items WHERE memory_id = ?",
                        (memory_id,),
                    ).fetchone()
                else:
                    # Only return valid (non-invalidated) memories
                    row = conn.execute(
                        "SELECT * FROM memory_items WHERE memory_id = ? AND invalidated_at IS NULL",
                        (memory_id,),
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

    def count_memories(self, include_invalidated: bool = False) -> int:
        """Count memories in storage."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                if include_invalidated:
                    row = conn.execute(
                        "SELECT COUNT(*) FROM memory_items"
                    ).fetchone()
                else:
                    row = conn.execute(
                        "SELECT COUNT(*) FROM memory_items WHERE invalidated_at IS NULL"
                    ).fetchone()
            return int(row[0]) if row else 0
        except sqlite3.Error as e:
            raise StorageError(f"Failed to count memories: {e}")

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
                        tags=?, links=?, metadata=?, state_checksum=?,
                        invalidated_at=?, invalidation_reason=?
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
                        memory.state_checksum,
                        memory.invalidated_at,
                        memory.invalidation_reason,
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
        include_invalidated: bool = False,
    ) -> List[MemoryItem]:
        """Keyword + tag search.

        Args:
            query: Text search query
            tags: Filter by tags
            limit: Maximum results
            include_invalidated: If False, excludes invalidated memories

        Returns:
            List of matching MemoryItem objects
        """
        try:
            sql = "SELECT * FROM memory_items WHERE 1=1"
            params: list = []

            # Exclude invalidated by default
            if not include_invalidated:
                sql += " AND invalidated_at IS NULL"

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

    def get_all(
        self,
        limit: int = 100,
        offset: int = 0,
        include_invalidated: bool = False,
    ) -> List[MemoryItem]:
        """Return all memories up to limit, with optional offset for pagination.

        Args:
            limit: Maximum number of memories to return
            offset: Number of memories to skip (for pagination)
            include_invalidated: If False, excludes invalidated memories

        Returns:
            List of MemoryItem objects
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                if include_invalidated:
                    rows = conn.execute(
                        "SELECT * FROM memory_items ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                        (int(limit), int(offset)),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM memory_items WHERE invalidated_at IS NULL ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                        (int(limit), int(offset)),
                    ).fetchall()
            return [self._row_to_memory(row) for row in rows]
        except sqlite3.Error as e:
            raise StorageError(f"Failed to get memories: {e}")

    def semantic_search(
        self,
        query: str,
        limit: int = 10,
        threshold: float = 0.5,
        include_invalidated: bool = False,
    ) -> List[Tuple[MemoryItem, float]]:
        """Search memories using semantic similarity.

        Uses embeddings to find memories similar to the query.
        Falls back to keyword search if no embeddings exist.

        Args:
            query: Search query text
            limit: Maximum results to return
            threshold: Minimum similarity score (0.0-1.0)
            include_invalidated: If False, excludes invalidated memories

        Returns:
            List of (memory, similarity_score) tuples, sorted by score
        """
        try:
            # Compute query embedding
            query_embedding = _compute_embedding(query)

            # Get all memories with embeddings
            with sqlite3.connect(self.db_path) as conn:
                if include_invalidated:
                    rows = conn.execute(
                        "SELECT memory_id, embedding FROM memory_embeddings"
                    ).fetchall()
                else:
                    # Join with memory_items to filter out invalidated
                    rows = conn.execute(
                        """SELECT e.memory_id, e.embedding
                           FROM memory_embeddings e
                           JOIN memory_items m ON e.memory_id = m.memory_id
                           WHERE m.invalidated_at IS NULL"""
                    ).fetchall()

            # Compute similarities
            results = []
            for memory_id, embedding_blob in rows:
                similarity = _cosine_similarity(query_embedding, embedding_blob)
                if similarity >= threshold:
                    memory = self.get_memory(memory_id, include_invalidated=include_invalidated)
                    if memory:
                        results.append((memory, similarity))

            # Sort by similarity descending
            results.sort(key=lambda x: x[1], reverse=True)

            # Return top results
            return results[:limit]

        except sqlite3.Error as e:
            # Fall back to keyword search on error
            memories = self.search_memories(query=query, limit=limit, include_invalidated=include_invalidated)
            return [(m, 0.5) for m in memories]

    def save_embedding(self, memory_id: str, content: str) -> None:
        """Compute and save embedding for a memory.

        Args:
            memory_id: Memory ID to save embedding for
            content: Content to embed
        """
        try:
            embedding = _compute_embedding(content)
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO memory_embeddings
                       (memory_id, embedding, created_at)
                       VALUES (?, ?, ?)""",
                    (memory_id, embedding, time()),
                )
                conn.commit()
        except sqlite3.Error as e:
            raise StorageError(f"Failed to save embedding: {e}")

    # ------------------------------------------------------------------
    # Checksum and State Validation
    # ------------------------------------------------------------------

    def _compute_state_checksum(self, memory: MemoryItem) -> str:
        """Compute checksum of memory state for corruption detection."""
        state_dict = {
            "memory_id": memory.memory_id,
            "content_hash": memory.content_hash,
            "salience": round(memory.salience, 6),
            "confidence": round(memory.confidence, 6),
            "volatility": round(memory.volatility, 6),
            "stability": round(memory.stability, 6),
            "decay_rate": round(memory.decay_rate, 6),
            "legal_salience": round(memory.legal_salience, 6),
            "trust_salience": round(memory.trust_salience, 6),
            "interest_salience": round(memory.interest_salience, 6),
            "attention_salience": round(memory.attention_salience, 6),
            "access_count": memory.access_count,
            "reinforcement_count": memory.reinforcement_count,
            "contradiction_count": memory.contradiction_count,
            "invalidated_at": memory.invalidated_at,
            "invalidation_reason": memory.invalidation_reason,
            "source_refs": sorted(memory.source_refs),
            "tags": sorted(memory.tags),
            "metadata": memory.metadata,
        }
        state_json = json.dumps(state_dict, sort_keys=True)
        return hashlib.sha256(state_json.encode()).hexdigest()[:16]

    def update_checksum(self, memory_id: str) -> str:
        """Compute and store checksum for a memory."""
        # Include invalidated memories since invalidation updates checksum
        memory = self.get_memory(memory_id, include_invalidated=True)
        if not memory:
            raise StorageError(f"Memory not found: {memory_id}")

        checksum = self._compute_state_checksum(memory)
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "UPDATE memory_items SET state_checksum = ? WHERE memory_id = ?",
                    (checksum, memory_id),
                )
                conn.commit()
            return checksum
        except sqlite3.Error as e:
            raise StorageError(f"Failed to update checksum: {e}")

    def verify_checksum(self, memory_id: str) -> bool:
        """Verify stored checksum matches computed checksum."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT * FROM memory_items WHERE memory_id = ?",
                    (memory_id,),
                ).fetchone()

            if not row or len(row) <= 22 or not row[22]:
                return True  # No checksum stored, skip validation

            stored_checksum = row[22]
            memory = self._row_to_memory(row)
            computed_checksum = self._compute_state_checksum(memory)
            return stored_checksum == computed_checksum
        except sqlite3.Error as e:
            raise StorageError(f"Failed to verify checksum: {e}")

    def verify_all_checksums(self) -> Dict[str, Any]:
        """Verify all stored checksums and return report."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT * FROM memory_items WHERE state_checksum IS NOT NULL"
                ).fetchall()

            results: Dict[str, Any] = {"valid": 0, "invalid": 0, "total": len(rows), "errors": []}
            for row in rows:
                memory = self._row_to_memory(row)
                stored_checksum: Optional[str] = row[22] if len(row) > 22 else None
                computed_checksum: str = self._compute_state_checksum(memory)
                if stored_checksum == computed_checksum:
                    results["valid"] = int(results["valid"]) + 1
                else:
                    results["invalid"] = int(results["invalid"]) + 1
                    error_list: List[Dict[str, Any]] = results["errors"]
                    error_list.append({
                        "memory_id": memory.memory_id,
                        "stored": stored_checksum,
                        "computed": computed_checksum,
                    })
            return results
        except sqlite3.Error as e:
            raise StorageError(f"Failed to verify checksums: {e}")

    # ------------------------------------------------------------------
    # Invalidation
    # ------------------------------------------------------------------

    def invalidate(
        self,
        memory_id: str,
        reason: str = "",
        timestamp: Optional[float] = None,
    ) -> bool:
        """Mark a memory as invalidated (logically deleted but preserved).

        Args:
            memory_id: Memory to invalidate
            reason: Why the memory was invalidated
            timestamp: When invalidated (default: current time)

        Returns:
            True if invalidated, False if not found
        """
        from time import time

        try:
            with sqlite3.connect(self.db_path) as conn:
                result = conn.execute(
                    """UPDATE memory_items
                       SET invalidated_at = ?, invalidation_reason = ?
                       WHERE memory_id = ?""",
                    (timestamp or time(), reason, memory_id),
                )
                conn.commit()
                return result.rowcount > 0
        except sqlite3.Error as e:
            raise StorageError(f"Failed to invalidate memory: {e}")

    def is_invalidated(self, memory_id: str) -> bool:
        """Check if a memory has been invalidated."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT invalidated_at FROM memory_items WHERE memory_id = ?",
                    (memory_id,),
                ).fetchone()
                return row is not None and row[0] is not None
        except sqlite3.Error as e:
            raise StorageError(f"Failed to check invalidation: {e}")

    def get_invalidated_memories(self) -> List[MemoryItem]:
        """Return all invalidated memories."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT * FROM memory_items WHERE invalidated_at IS NOT NULL"
                ).fetchall()
            return [self._row_to_memory(row) for row in rows]
        except sqlite3.Error as e:
            raise StorageError(f"Failed to get invalidated memories: {e}")

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

    # ------------------------------------------------------------------
    # Event Log Export / Rebuild
    # ------------------------------------------------------------------

    def export_all_events(self) -> List[MemoryEvent]:
        """Export all events from the event log for backup/rebuild."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT * FROM memory_events ORDER BY timestamp"
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
            raise StorageError(f"Failed to export events: {e}")

    def export_events_since(self, since_timestamp: float) -> List[MemoryEvent]:
        """Export events since a given timestamp for incremental rebuild."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT * FROM memory_events WHERE timestamp >= ? ORDER BY timestamp",
                    (since_timestamp,),
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
            raise StorageError(f"Failed to export events: {e}")

    def clear_all_memories(self) -> None:
        """Clear all memory items (for rebuild). Preserves events."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM memory_items")
                conn.execute("DELETE FROM memory_links")
                conn.commit()
        except sqlite3.Error as e:
            raise StorageError(f"Failed to clear memories: {e}")


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
