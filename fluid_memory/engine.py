"""Fluid Memory Engine

The main entry point for fluid memory operations.
"""

import hashlib
import uuid
from time import time
from typing import Optional, List, Dict, Any

from fluid_memory.config import FluidMemoryConfig
from fluid_memory.models import MemoryItem, MemoryLink, RetrievalResult
from fluid_memory.storage import MemoryStorage
from fluid_memory.decay import DecayManager, apply_decay_to_memory
from fluid_memory.events import MemoryEvent, EventType
from fluid_memory.mutation import mutate_memory
from fluid_memory.scoring import compute_retrieval_score
from fluid_memory.exceptions import MemoryNotFoundError, DuplicateMemoryError, StorageError


def _sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _new_event_id() -> str:
    return f"evt_{uuid.uuid4().hex[:16]}"


class FluidMemoryEngine:
    """Main fluid memory engine.

    Manages memory items with dynamic state properties:
    - Salience (importance)
    - Confidence (reliability)
    - Decay over time
    - Reinforcement and contradiction tracking
    - Append-only event log
    """

    def __init__(self, config: FluidMemoryConfig):
        self.config = config
        self.storage = MemoryStorage(str(config.db_path))
        self.decay = DecayManager(config)

    # ------------------------------------------------------------------
    # Add / get
    # ------------------------------------------------------------------

    def add_memory(
        self,
        content: str,
        tags: Optional[List[str]] = None,
        source_refs: Optional[List[str]] = None,
        salience: float = 0.5,
        confidence: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryItem:
        """Add a new memory item.

        Raises:
            DuplicateMemoryError: If identical content already exists.
        """
        content_hash = _sha256(content)

        existing = self.storage.get_memory_by_hash(content_hash)
        if existing:
            raise DuplicateMemoryError(
                f"Memory with this content already exists: {existing.memory_id}"
            )

        memory = MemoryItem(
            content=content,
            content_hash=content_hash,
            salience=salience,
            confidence=confidence,
            legal_salience=salience,
            trust_salience=salience,
            interest_salience=salience,
            attention_salience=salience,
            tags=tags or [],
            source_refs=source_refs or [],
            metadata=metadata or {},
        )

        self.storage.save_memory(memory)
        self._emit(memory.memory_id, EventType.CREATED, {})
        return memory

    def get_memory(self, memory_id: str) -> MemoryItem:
        """Get a memory by ID, recording an access event.

        Raises:
            MemoryNotFoundError: If memory_id not found.
        """
        memory = self.storage.get_memory(memory_id)
        if not memory:
            raise MemoryNotFoundError(f"Memory not found: {memory_id}")
        memory.touch()
        self.storage.update_memory(memory)
        self._emit(memory.memory_id, EventType.ACCESSED, {})
        return memory

    # ------------------------------------------------------------------
    # Reinforce / contradict
    # ------------------------------------------------------------------

    def reinforce(
        self,
        memory_id: str,
        amount: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryItem:
        """Increase salience and confidence.

        Raises:
            MemoryNotFoundError: If memory_id not found.
        """
        amount = amount if amount is not None else self.config.reinforcement_boost

        memory = self.storage.get_memory(memory_id)
        if not memory:
            raise MemoryNotFoundError(f"Memory not found: {memory_id}")

        old = {"salience": memory.salience, "confidence": memory.confidence}
        memory.salience = min(1.0, memory.salience + amount)
        memory.confidence = min(1.0, memory.confidence + (amount * 0.5))
        memory.reinforcement_count += 1
        memory.updated_at = time()
        new = {"salience": memory.salience, "confidence": memory.confidence}

        self.storage.update_memory(memory)
        self._emit(
            memory.memory_id,
            EventType.REINFORCED,
            {"old": old, "new": new},
            metadata=metadata,
        )
        return memory

    def contradict(
        self,
        memory_id: str,
        amount: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryItem:
        """Decrease confidence.

        Raises:
            MemoryNotFoundError: If memory_id not found.
        """
        amount = amount if amount is not None else self.config.contradiction_penalty

        memory = self.storage.get_memory(memory_id)
        if not memory:
            raise MemoryNotFoundError(f"Memory not found: {memory_id}")

        old = {"salience": memory.salience, "confidence": memory.confidence}
        memory.confidence = max(0.0, memory.confidence - amount)
        memory.contradiction_count += 1
        memory.updated_at = time()
        new = {"salience": memory.salience, "confidence": memory.confidence}

        self.storage.update_memory(memory)
        self._emit(
            memory.memory_id,
            EventType.CONTRADICTED,
            {"old": old, "new": new},
            metadata=metadata,
        )
        return memory

    # ------------------------------------------------------------------
    # Mutate / link
    # ------------------------------------------------------------------

    def mutate(
        self,
        memory_id: str,
        new_content: Optional[str] = None,
        state_delta: Optional[Dict[str, Any]] = None,
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryItem:
        """Perform controlled mutation on a memory.

        Raises:
            MemoryNotFoundError: If memory_id not found.
        """
        memory = self.storage.get_memory(memory_id)
        if not memory:
            raise MemoryNotFoundError(f"Memory not found: {memory_id}")

        memory, event = mutate_memory(
            memory,
            new_content=new_content,
            state_delta=state_delta,
            reason=reason,
            metadata=metadata,
        )
        self.storage.update_memory(memory)
        self.storage.save_event(event)
        return memory

    def link_memories(
        self,
        source_id: str,
        target_id: str,
        link_type: str = "related",
        strength: float = 0.5,
    ) -> MemoryLink:
        """Create a directed link between two memories.

        Raises:
            MemoryNotFoundError: If either memory_id is not found.
        """
        if not self.storage.get_memory(source_id):
            raise MemoryNotFoundError(f"Source memory not found: {source_id}")
        if not self.storage.get_memory(target_id):
            raise MemoryNotFoundError(f"Target memory not found: {target_id}")

        link = MemoryLink(
            source_memory_id=source_id,
            target_memory_id=target_id,
            link_type=link_type,
            strength=strength,
        )
        self.storage.save_link(link)
        self._emit(
            source_id,
            EventType.LINKED,
            {"target_id": target_id, "link_type": link_type, "strength": strength},
        )
        return link

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> List[RetrievalResult]:
        """Retrieve memories matching query and/or tags."""
        limit = limit or self.config.max_results
        threshold = threshold or self.config.retrieval_threshold

        memories = self.storage.search_memories(query=query, tags=tags, limit=limit)

        results = []
        for memory in memories:
            memory.touch()
            self.storage.update_memory(memory)
            score = compute_retrieval_score(memory, query=query, tags=tags)
            if score >= threshold:
                results.append(RetrievalResult(
                    memory=memory,
                    score=score,
                    match_type="keyword",
                ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    # ------------------------------------------------------------------
    # Decay
    # ------------------------------------------------------------------

    def apply_decay(self, days: float = 1.0) -> int:
        """Apply decay to all memories.

        Returns:
            Number of memories decayed.
        """
        memories = self.storage.get_all(limit=1000)
        count = 0
        for memory in memories:
            old_salience = memory.salience
            self.decay.apply_all_decay(memory, days)
            self.storage.update_memory(memory)
            if memory.salience < old_salience:
                self._emit(
                    memory.memory_id,
                    EventType.DECAYED,
                    {"old_salience": old_salience, "new_salience": memory.salience},
                )
            count += 1
        return count

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def get_events(self, memory_id: str) -> List[MemoryEvent]:
        """Return all events for a memory, oldest first."""
        return self.storage.get_events(memory_id)

    # ------------------------------------------------------------------
    # Stats / close
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        memories = self.storage.get_all(limit=10000)

        if not memories:
            return {
                "total_memories": 0,
                "avg_salience": 0.0,
                "avg_confidence": 0.0,
            }

        return {
            "total_memories": len(memories),
            "avg_salience": sum(m.salience for m in memories) / len(memories),
            "avg_confidence": sum(m.confidence for m in memories) / len(memories),
            "total_reinforcements": sum(m.reinforcement_count for m in memories),
            "total_contradictions": sum(m.contradiction_count for m in memories),
        }

    def close(self) -> None:
        """Close the engine (no-op for SQLite, but good practice)."""
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _emit(
        self,
        memory_id: str,
        event_type: EventType,
        delta: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        event = MemoryEvent(
            event_id=_new_event_id(),
            memory_id=memory_id,
            event_type=event_type,
            delta_json=delta,
            metadata_json=metadata or {},
        )
        self.storage.save_event(event)
