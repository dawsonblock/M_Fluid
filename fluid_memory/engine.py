"""Fluid Memory Engine

The main entry point for fluid memory operations.
"""

import hashlib
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any

from fluid_memory.config import FluidMemoryConfig
from fluid_memory.models import MemoryItem, RetrievalResult
from fluid_memory.storage import FluidMemoryStorage, compute_sha256
from fluid_memory.decay import DecayManager
from fluid_memory.exceptions import MemoryNotFoundError, StorageError


class FluidMemoryEngine:
    """Main fluid memory engine.

    Manages memory items with dynamic state properties:
    - Salience (importance)
    - Confidence (reliability)
    - Decay over time
    - Reinforcement and contradiction tracking
    """

    def __init__(self, config: FluidMemoryConfig):
        self.config = config
        self.storage = FluidMemoryStorage(config)
        self.decay = DecayManager(config)

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

        Args:
            content: The memory content
            tags: Optional categorization tags
            source_refs: Optional source references
            salience: Initial salience (0.0-1.0)
            confidence: Initial confidence (0.0-1.0)
            metadata: Optional additional metadata

        Returns:
            The created MemoryItem
        """
        content_hash = compute_sha256(content)

        # Check for duplicates
        existing = self.storage.get_by_hash(content_hash)
        if existing:
            # Touch existing memory
            existing.touch()
            self.storage.store(existing)
            return existing

        # Create new memory
        memory = MemoryItem(
            memory_id=f"mem_{uuid.uuid4().hex[:16]}",
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

        return self.storage.store(memory)

    def retrieve(
        self,
        query: str,
        limit: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> List[RetrievalResult]:
        """Retrieve memories matching query.

        Uses keyword search with fluid scoring (salience * confidence).

        Args:
            query: Search query
            limit: Maximum results (default from config)
            threshold: Minimum score threshold (default from config)

        Returns:
            List of retrieval results sorted by score
        """
        limit = limit or self.config.max_results
        threshold = threshold or self.config.retrieval_threshold

        # Search memories
        memories = self.storage.search(query, limit=limit)

        results = []
        for memory in memories:
            # Touch memory on retrieval
            memory.touch()
            self.storage.store(memory)

            # Calculate fluid score
            score = memory.salience * memory.confidence

            if score >= threshold:
                results.append(RetrievalResult(
                    memory=memory,
                    score=score,
                    match_type="keyword",
                ))

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    def get_by_id(self, memory_id: str) -> Optional[MemoryItem]:
        """Get a memory by ID."""
        memory = self.storage.get_by_id(memory_id)
        if memory:
            memory.touch()
            self.storage.store(memory)
        return memory

    def reinforce(self, memory_id: str, amount: Optional[float] = None) -> MemoryItem:
        """Reinforce a memory (increase salience and confidence).

        Args:
            memory_id: The memory to reinforce
            amount: Reinforcement amount (default from config)

        Returns:
            The updated MemoryItem

        Raises:
            MemoryNotFoundError: If memory_id not found
        """
        amount = amount or self.config.reinforcement_boost

        memory = self.storage.get_by_id(memory_id)
        if not memory:
            raise MemoryNotFoundError(f"Memory not found: {memory_id}")

        # Increase salience and confidence
        memory.salience = min(1.0, memory.salience + amount)
        memory.confidence = min(1.0, memory.confidence + (amount * 0.5))
        memory.reinforcement_count += 1
        memory.updated_at = datetime.utcnow()

        return self.storage.store(memory)

    def contradict(self, memory_id: str, penalty: Optional[float] = None) -> MemoryItem:
        """Mark a memory as contradicted (decrease confidence).

        Args:
            memory_id: The memory to contradict
            penalty: Confidence penalty (default from config)

        Returns:
            The updated MemoryItem

        Raises:
            MemoryNotFoundError: If memory_id not found
        """
        penalty = penalty or self.config.contradiction_penalty

        memory = self.storage.get_by_id(memory_id)
        if not memory:
            raise MemoryNotFoundError(f"Memory not found: {memory_id}")

        # Decrease confidence
        memory.confidence = max(0.0, memory.confidence - penalty)
        memory.contradiction_count += 1
        memory.updated_at = datetime.utcnow()

        return self.storage.store(memory)

    def apply_decay(self, days: float = 1.0) -> int:
        """Apply decay to all memories.

        Args:
            days: Number of days to decay

        Returns:
            Number of memories decayed
        """
        memories = self.storage.get_all(limit=1000)

        count = 0
        for memory in memories:
            self.decay.apply_all_decay(memory, days)
            self.storage.store(memory)
            count += 1

        return count

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
