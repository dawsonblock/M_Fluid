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
from fluid_memory.decay import DecayManager
from fluid_memory.events import MemoryEvent, EventType
from fluid_memory.mutation import mutate_memory
from fluid_memory.scoring import compute_retrieval_score
from fluid_memory.exceptions import MemoryNotFoundError, DuplicateMemoryError
from fluid_memory.audit_logger import AuditLogger, NoOpAuditLogger
from fluid_memory.health import FluidMemoryHealth
from fluid_memory.metrics import FluidMemoryMetrics
from fluid_memory.batch import BatchOperations


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

    def __init__(self, config: FluidMemoryConfig, enable_audit: bool = True):
        self.config = config
        self.storage = MemoryStorage(str(config.db_path))
        self.decay = DecayManager(config)
        # Use NoOpAuditLogger when disabled for true silence
        if enable_audit:
            self.audit = AuditLogger(enable_file=True)
        else:
            self.audit = NoOpAuditLogger()
        self.health = FluidMemoryHealth(self)
        self.metrics = FluidMemoryMetrics()
        self.batch = BatchOperations(self)

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
        volatility: float = 0.3,
        metadata: Optional[Dict[str, Any]] = None,
        detect_contradictions: bool = True,
        contradiction_threshold: float = 0.7,
    ) -> MemoryItem:
        """Add a new memory item.

        Args:
            content: Memory content
            tags: Optional tags
            source_refs: Optional source references
            salience: Initial salience
            confidence: Initial confidence
            volatility: Initial volatility
            metadata: Optional metadata
            detect_contradictions: If True, check for potential contradictions
            contradiction_threshold: Similarity threshold for contradiction detection

        Raises:
            DuplicateMemoryError: If identical content already exists.

        Returns:
            Created MemoryItem
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
            volatility=volatility,
            legal_salience=salience,
            trust_salience=salience,
            interest_salience=salience,
            attention_salience=salience,
            tags=tags or [],
            source_refs=source_refs or [],
            metadata=metadata or {},
        )

        self.storage.save_memory(memory)
        self.storage.save_embedding(memory.memory_id, content)
        self._emit(memory.memory_id, EventType.CREATED, {})
        self.audit.log_memory_created(
            memory_id=memory.memory_id,
            content_hash=content_hash,
            tags=tags or [],
            metadata=metadata or {},
        )

        # Detect and record potential contradictions
        if detect_contradictions:
            self._detect_and_record_contradictions(
                memory, threshold=contradiction_threshold
            )

        # Update checksum after creation
        self.storage.update_checksum(memory.memory_id)

        return memory

    def _detect_and_record_contradictions(
        self,
        new_memory: MemoryItem,
        threshold: float = 0.7,
    ) -> List[str]:
        """Detect potentially contradictory memories and record them.

        Uses semantic similarity to find memories that might contradict
        the new memory. Similar memories with opposing sentiment or
        conflicting information are flagged.

        Args:
            new_memory: The newly added memory
            threshold: Similarity threshold for potential contradiction

        Returns:
            List of contradictory memory IDs
        """
        contradictory_ids = []

        try:
            # Search for semantically similar memories
            similar = self.storage.semantic_search(
                new_memory.content,
                limit=10,
                threshold=threshold,
                include_invalidated=False,
            )

            for similar_memory, similarity in similar:
                # Skip the memory itself
                if similar_memory.memory_id == new_memory.memory_id:
                    continue

                # Check for contradiction indicators
                # Simple heuristic: different confidence levels or explicit contradiction tags
                is_contradictory = False
                contradiction_reason = None

                # Check for contradiction tags
                if "contradiction" in similar_memory.tags or "disputed" in similar_memory.tags:
                    is_contradictory = True
                    contradiction_reason = "Tagged as contradictory"

                # Check for significant confidence difference
                confidence_diff = abs(new_memory.confidence - similar_memory.confidence)
                if confidence_diff > 0.3:
                    is_contradictory = True
                    contradiction_reason = f"Confidence mismatch ({confidence_diff:.2f})"

                # If potentially contradictory, apply contradiction pressure
                if is_contradictory:
                    contradictory_ids.append(similar_memory.memory_id)

                    # Apply contradiction to both memories
                    self.contradict(
                        similar_memory.memory_id,
                        amount=0.1,
                        conflicting_memory_id=new_memory.memory_id,
                        reason=contradiction_reason or "Detected similar but potentially conflicting content",
                    )
                    self.contradict(
                        new_memory.memory_id,
                        amount=0.1,
                        conflicting_memory_id=similar_memory.memory_id,
                        reason=contradiction_reason or "Detected similar but potentially conflicting content",
                    )

            # Record contradiction metadata
            if contradictory_ids:
                new_memory.metadata["detected_contradictions"] = contradictory_ids
                self.storage.update_memory(new_memory)

        except Exception:
            # Fail silently - contradiction detection is best-effort
            pass

        return contradictory_ids

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
        # Access mutates access_count, last_accessed_at, and updated_at.
        # Keep checksum valid after read-touch mutation.
        self.storage.update_checksum(memory.memory_id)
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
        # Update checksum after reinforcement
        self.storage.update_checksum(memory.memory_id)
        return memory

    def contradict(
        self,
        memory_id: str,
        amount: Optional[float] = None,
        conflicting_memory_id: Optional[str] = None,
        conflicting_evidence_id: Optional[str] = None,
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryItem:
        """Apply contradiction pressure to a memory.

        When contradiction is detected/applied, this method:
        - Lowers confidence
        - Increases volatility
        - Lowers stability
        - Increases review salience (attention_salience)
        - Increments contradiction_count
        - Records contradiction event with link to conflicting source

        Args:
            memory_id: Memory to contradict
            amount: Contradiction penalty amount (default: config.contradiction_penalty)
            conflicting_memory_id: ID of the memory that contradicts this one
            conflicting_evidence_id: ID of the evidence that contradicts this memory
            reason: Explanation for the contradiction
            metadata: Additional metadata

        Raises:
            MemoryNotFoundError: If memory_id not found.

        Returns:
            Updated MemoryItem
        """
        amount = amount if amount is not None else self.config.contradiction_penalty

        memory = self.storage.get_memory(memory_id)
        if not memory:
            raise MemoryNotFoundError(f"Memory not found: {memory_id}")

        # Record old state
        old = {
            "salience": memory.salience,
            "confidence": memory.confidence,
            "volatility": memory.volatility,
            "stability": memory.stability,
            "attention_salience": memory.attention_salience,
        }

        # Apply contradiction state changes
        # 1. Lower confidence
        memory.confidence = max(0.0, memory.confidence - amount)

        # 2. Increase volatility (memory becomes more changeable)
        memory.volatility = min(1.0, memory.volatility + (amount * 0.5))

        # 3. Lower stability (memory becomes less stable)
        memory.stability = max(0.0, memory.stability - (amount * 0.3))

        # 4. Increase review salience (needs attention)
        memory.attention_salience = min(1.0, memory.attention_salience + (amount * 0.8))

        # 5. Increment contradiction count
        memory.contradiction_count += 1

        # 6. Update timestamp
        memory.updated_at = time()

        # Build new state record
        new = {
            "salience": memory.salience,
            "confidence": memory.confidence,
            "volatility": memory.volatility,
            "stability": memory.stability,
            "attention_salience": memory.attention_salience,
        }

        self.storage.update_memory(memory)

        # Build event metadata with contradiction links
        event_metadata = {
            "old": old,
            "new": new,
            "reason": reason,
        }
        if conflicting_memory_id:
            event_metadata["conflicting_memory_id"] = conflicting_memory_id
        if conflicting_evidence_id:
            event_metadata["conflicting_evidence_id"] = conflicting_evidence_id
        if metadata:
            event_metadata.update(metadata)

        self._emit(
            memory.memory_id,
            EventType.CONTRADICTED,
            event_metadata,
        )

        # Audit log the contradiction
        if conflicting_memory_id:
            self.audit.log_contradiction_applied(
                memory_id=memory.memory_id,
                conflicting_memory_id=conflicting_memory_id,
                amount=amount,
                old_confidence=old["confidence"],
                new_confidence=new["confidence"],
            )

        # Update checksum after contradiction
        self.storage.update_checksum(memory.memory_id)

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
        # Update checksum after mutation
        self.storage.update_checksum(memory.memory_id)
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
        use_semantic: bool = False,
        semantic_threshold: float = 0.5,
        enable_temporal_boost: bool = True,
        enable_deduplication: bool = True,
        enable_mmr: bool = False,
        mmr_lambda: float = 0.5,
    ) -> List[RetrievalResult]:
        """Retrieve memories matching query and/or tags.

        Args:
            query: Search query text
            tags: Filter by tags
            limit: Maximum results
            threshold: Minimum retrieval score
            use_semantic: If True, use semantic search (embeddings)
            semantic_threshold: Minimum similarity for semantic search
            enable_temporal_boost: Weight recent memories higher
            enable_deduplication: Remove near-duplicate results
            enable_mmr: Use Maximal Marginal Relevance for diversity
            mmr_lambda: MMR trade-off (0=diversity, 1=relevance)

        Returns:
            List of RetrievalResult objects
        """
        from fluid_memory.retrieval import (
            temporal_boost,
            remove_duplicates,
            maximal_marginal_relevance,
        )

        limit = limit or self.config.max_results
        threshold = threshold or self.config.retrieval_threshold

        if use_semantic and query:
            # Semantic search path
            semantic_hits = self.storage.semantic_search(
                query=query,
                limit=limit * 3,
                threshold=semantic_threshold,
                include_invalidated=False,
            )
            results = []
            current_time = time()
            for memory, similarity in semantic_hits:
                score = compute_retrieval_score(
                    memory, query=query, tags=tags
                ) * similarity
                if enable_temporal_boost:
                    score *= temporal_boost(memory, current_time)
                results.append(
                    RetrievalResult(
                        memory=memory,
                        score=score,
                        match_type="semantic",
                    )
                )
            results.sort(key=lambda r: r.score, reverse=True)
            if enable_deduplication:
                results = remove_duplicates(results)
            if enable_mmr:
                results = maximal_marginal_relevance(
                    results,
                    query,
                    lambda_param=mmr_lambda,
                    limit=limit,
                )
            results = results[:limit]
        else:
            # Keyword search path with enhanced retrieval
            from fluid_memory.retrieval import retrieve as retrieve_fn

            results = retrieve_fn(
                storage=self.storage,
                query=query,
                tags=tags,
                limit=limit,
                enable_temporal_boost=enable_temporal_boost,
                enable_deduplication=enable_deduplication,
                enable_mmr=enable_mmr,
                mmr_lambda=mmr_lambda,
            )

        # Apply threshold filter and touch/access logging
        filtered_results = []
        for result in results:
            if result.score >= threshold:
                result.memory.touch()
                self.storage.update_memory(result.memory)
                self.storage.update_checksum(result.memory.memory_id)
                self._emit(result.memory.memory_id, EventType.ACCESSED, {})
                filtered_results.append(result)

        # Log search
        if query and filtered_results:
            self.audit.log_memory_accessed(
                memory_id=filtered_results[0].memory.memory_id,
                access_type="semantic_search" if use_semantic else "search",
            )

        return filtered_results

    # ------------------------------------------------------------------
    # Decay
    # ------------------------------------------------------------------

    def apply_decay(self, days: float = 1.0, batch_size: int = 1000) -> int:
        """Apply decay to all active memories.

        Args:
            days: Number of days to simulate decay for
            batch_size: Number of memories to process per batch

        Returns:
            Number of memories decayed.
        """
        count = 0
        offset = 0
        while True:
            memories = self.storage.get_all(limit=batch_size, offset=offset)
            if not memories:
                break
            for memory in memories:
                old_salience = memory.salience
                self.decay.apply_all_decay(memory, days)
                self.storage.update_memory(memory)
                # Update checksum after decay
                self.storage.update_checksum(memory.memory_id)
                if memory.salience < old_salience:
                    self._emit(
                        memory.memory_id,
                        EventType.DECAYED,
                        {"old_salience": old_salience, "new_salience": memory.salience},
                    )
                count += 1
            offset += batch_size
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

    def invalidate_memory(
        self,
        memory_id: str,
        reason: str = "",
    ) -> MemoryItem:
        """Invalidate a memory (soft delete).

        Args:
            memory_id: Memory to invalidate
            reason: Explanation for invalidation

        Returns:
            Invalidated MemoryItem

        Raises:
            MemoryNotFoundError: If memory_id not found.
        """
        # Get memory before invalidating
        memory = self.storage.get_memory(memory_id, include_invalidated=True)
        if not memory:
            from fluid_memory.exceptions import MemoryNotFoundError
            raise MemoryNotFoundError(f"Memory not found: {memory_id}")

        # Invalidate in storage
        success = self.storage.invalidate(memory_id, reason=reason)
        if not success:
            from fluid_memory.exceptions import MemoryNotFoundError
            raise MemoryNotFoundError(f"Failed to invalidate memory: {memory_id}")

        # Update memory object to reflect invalidation
        from time import time
        memory.invalidated_at = time()
        memory.invalidation_reason = reason

        self._emit(
            memory_id,
            EventType.INVALIDATED,
            {"reason": reason},
        )
        # Invalidation mutates invalidated_at and invalidation_reason.
        # Keep checksum valid after soft-delete mutation.
        self.storage.update_checksum(memory_id)
        self.audit.log_memory_invalidated(
            memory_id=memory_id,
            reason=reason,
        )
        return memory

    def verify_memory(self, memory_id: str) -> bool:
        """Verify checksum of a single memory.

        Args:
            memory_id: Memory to verify

        Returns:
            True if checksum valid, False otherwise

        Raises:
            MemoryNotFoundError: If memory_id not found.
        """
        return self.storage.verify_checksum(memory_id)

    def verify_all_memory_checksums(self) -> Dict[str, Any]:
        """Verify checksums of all memories.

        Returns:
            Dict with counts of valid/invalid memories and list of invalid IDs
        """
        return self.storage.verify_all_checksums()

    def create_orchestrator(self) -> "MemoryOrchestrator":
        """Create a MemoryOrchestrator for evidence-grounded retrieval.

        Returns:
            MemoryOrchestrator instance bound to this engine
        """
        from fluid_memory.orchestrator import MemoryOrchestrator
        return MemoryOrchestrator(self)

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
