"""
Fluid Memory Engine

Main orchestrator for the fluid memory system.
"""

from typing import Optional, List, Dict, Any
from time import time

from fluid_memory.config import FluidMemoryConfig
from fluid_memory.models import MemoryItem, RetrievalResult, MemoryLink
from fluid_memory.events import MemoryEvent, EventType
from fluid_memory.storage import MemoryStorage
from fluid_memory.scoring import compute_salience, compute_confidence
from fluid_memory.decay import apply_decay
from fluid_memory.mutation import mutate_memory
from fluid_memory.retrieval import retrieve
from fluid_memory.exceptions import MemoryNotFoundError, DuplicateMemoryError
from fluid_memory.state import clamp01


class FluidMemoryEngine:
    """
    Main engine for fluid memory operations.
    
    Provides a unified interface for:
    - Adding and retrieving memories
    - Reinforcing and contradicting memories
    - Mutating memory state
    - Linking memories
    - Applying decay
    - Event tracking
    
    Usage:
        config = FluidMemoryConfig(data_dir="/path/to/data")
        engine = FluidMemoryEngine(config)
        
        memory = engine.add_memory("Important fact", tags=["key"])
        results = engine.retrieve("important")
        engine.reinforce(memory.memory_id)
        
        engine.close()
    """
    
    def __init__(self, config: Optional[FluidMemoryConfig] = None):
        self.config = config or FluidMemoryConfig()
        self.storage = MemoryStorage(self.config.sqlite_path)
    
    def add_memory(
        self,
        content: str,
        tags: Optional[List[str]] = None,
        source_refs: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryItem:
        """
        Add a new memory.
        
        Args:
            content: Text content of the memory
            tags: Optional tags for categorization
            source_refs: Optional source references
            metadata: Optional additional metadata
            
        Returns:
            The created MemoryItem
        """
        # Check for duplicate by hash
        content_hash = MemoryItem(content=content).content_hash
        existing = self.storage.get_memory_by_hash(content_hash)
        if existing:
            raise DuplicateMemoryError(
                f"Memory with this content already exists: {existing.memory_id}"
            )
        
        # Create memory with defaults
        memory = MemoryItem(
            content=content,
            tags=tags or [],
            source_refs=source_refs or [],
            metadata=metadata or {},
            salience=self.config.default_salience,
            confidence=self.config.default_confidence,
            volatility=self.config.default_volatility,
            stability=self.config.default_stability,
            decay_rate=self.config.default_decay_rate,
        )
        
        # Save memory
        self.storage.save_memory(memory)
        
        # Record creation event
        event = MemoryEvent(
            event_id=str(hash(f"{memory.memory_id}_created_{time()}")),
            memory_id=memory.memory_id,
            event_type=EventType.CREATED,
            delta_json={"content_hash": memory.content_hash},
        )
        self.storage.save_event(event)
        
        return memory
    
    def get_memory(self, memory_id: str) -> MemoryItem:
        """
        Retrieve a memory by ID.
        
        Args:
            memory_id: Memory ID to retrieve
            
        Returns:
            The MemoryItem
            
        Raises:
            MemoryNotFoundError: If memory not found
        """
        memory = self.storage.get_memory(memory_id)
        if memory is None:
            raise MemoryNotFoundError(f"Memory not found: {memory_id}")
        
        # Record access
        memory.touch()
        self.storage.update_memory(memory)
        
        # Record access event
        event = MemoryEvent(
            event_id=str(hash(f"{memory_id}_access_{time()}")),
            memory_id=memory_id,
            event_type=EventType.ACCESSED,
            delta_json={"access_count": memory.access_count},
        )
        self.storage.save_event(event)
        
        return memory
    
    def retrieve(
        self,
        query: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[RetrievalResult]:
        """
        Retrieve memories with fluid scoring.
        
        Args:
            query: Text query to match
            tags: Tags to filter by
            limit: Maximum results to return
            
        Returns:
            List of RetrievalResult sorted by score
        """
        return retrieve(self.storage, query=query, tags=tags, limit=limit)
    
    def reinforce(
        self,
        memory_id: str,
        amount: float = 0.1,
        source_ref: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryItem:
        """
        Reinforce a memory (increases salience, confidence, stability).
        
        Args:
            memory_id: Memory to reinforce
            amount: Reinforcement amount
            source_ref: Optional source reference
            metadata: Optional metadata
            
        Returns:
            Updated MemoryItem
        """
        memory = self.get_memory(memory_id)
        
        old_values = {
            "salience": memory.salience,
            "confidence": memory.confidence,
            "stability": memory.stability,
            "volatility": memory.volatility,
        }
        
        # Apply reinforcement
        memory.salience = clamp01(memory.salience + amount)
        memory.confidence = clamp01(memory.confidence + amount * 0.5)
        memory.stability = clamp01(memory.stability + amount * 0.3)
        memory.volatility = clamp01(memory.volatility - amount * 0.2)
        memory.reinforcement_count += 1
        
        # Add source ref if provided
        if source_ref and source_ref not in memory.source_refs:
            memory.source_refs.append(source_ref)
        
        memory.update_timestamp()
        self.storage.update_memory(memory)
        
        # Record event
        event = MemoryEvent(
            event_id=str(hash(f"{memory_id}_reinforce_{time()}")),
            memory_id=memory_id,
            event_type=EventType.REINFORCED,
            delta_json={"old": old_values, "new": {
                "salience": memory.salience,
                "confidence": memory.confidence,
                "stability": memory.stability,
                "volatility": memory.volatility,
            }},
            metadata_json=metadata or {},
        )
        self.storage.save_event(event)
        
        return memory
    
    def contradict(
        self,
        memory_id: str,
        amount: float = 0.1,
        source_ref: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryItem:
        """
        Record a contradiction for a memory (decreases confidence, increases volatility).
        
        Args:
            memory_id: Memory to contradict
            amount: Contradiction amount
            source_ref: Optional source reference
            metadata: Optional metadata
            
        Returns:
            Updated MemoryItem
        """
        memory = self.get_memory(memory_id)
        
        old_values = {
            "confidence": memory.confidence,
            "volatility": memory.volatility,
            "stability": memory.stability,
            "salience": memory.salience,
        }
        
        # Apply contradiction effects
        memory.confidence = clamp01(memory.confidence - amount)
        memory.volatility = clamp01(memory.volatility + amount * 0.5)
        memory.stability = clamp01(memory.stability - amount * 0.3)
        # Contradictions may increase salience (attention-grabbing)
        memory.salience = clamp01(memory.salience + amount * 0.2)
        memory.contradiction_count += 1
        
        # Add source ref if provided
        if source_ref and source_ref not in memory.source_refs:
            memory.source_refs.append(source_ref)
        
        memory.update_timestamp()
        self.storage.update_memory(memory)
        
        # Record event
        event = MemoryEvent(
            event_id=str(hash(f"{memory_id}_contradict_{time()}")),
            memory_id=memory_id,
            event_type=EventType.CONTRADICTED,
            delta_json={"old": old_values, "new": {
                "confidence": memory.confidence,
                "volatility": memory.volatility,
                "stability": memory.stability,
                "salience": memory.salience,
            }},
            metadata_json=metadata or {},
        )
        self.storage.save_event(event)
        
        return memory
    
    def mutate(
        self,
        memory_id: str,
        new_content: Optional[str] = None,
        state_delta: Optional[Dict[str, Any]] = None,
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryItem:
        """
        Perform controlled mutation on a memory.
        
        Args:
            memory_id: Memory to mutate
            new_content: New content (if changing)
            state_delta: State field changes
            reason: Reason for mutation
            metadata: Optional metadata
            
        Returns:
            Updated MemoryItem
        """
        memory = self.get_memory(memory_id)
        
        mutated_memory, event = mutate_memory(
            memory,
            new_content=new_content,
            state_delta=state_delta,
            reason=reason,
            metadata=metadata,
            mutation_resistance_enabled=self.config.mutation_resistance_enabled,
        )
        
        self.storage.update_memory(mutated_memory)
        self.storage.save_event(event)
        
        return mutated_memory
    
    def link_memories(
        self,
        source_id: str,
        target_id: str,
        link_type: str = "related",
        strength: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryLink:
        """
        Create a link between two memories.
        
        Args:
            source_id: Source memory ID
            target_id: Target memory ID
            link_type: Type of link (related, supports, contradicts, parent, child, sequence)
            strength: Link strength [0.0, 1.0]
            metadata: Optional metadata
            
        Returns:
            Created MemoryLink
        """
        # Verify both memories exist
        self.get_memory(source_id)
        self.get_memory(target_id)
        
        # Create link
        link = MemoryLink(
            source_memory_id=source_id,
            target_memory_id=target_id,
            link_type=link_type,
            strength=strength,
            metadata=metadata or {},
        )
        
        self.storage.save_link(link)
        
        # Update memory links list
        source_memory = self.storage.get_memory(source_id)
        if target_id not in source_memory.links:
            source_memory.links.append(target_id)
            self.storage.update_memory(source_memory)
        
        # Record event on source memory
        event = MemoryEvent(
            event_id=str(hash(f"{source_id}_link_{time()}")),
            memory_id=source_id,
            event_type=EventType.LINKED,
            delta_json={
                "target_id": target_id,
                "link_type": link_type,
                "strength": strength,
            },
        )
        self.storage.save_event(event)
        
        return link
    
    def apply_decay(
        self,
        now: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> int:
        """
        Apply decay to all memories.
        
        Args:
            now: Current timestamp (defaults to time.time())
            limit: Maximum memories to process
            
        Returns:
            Number of memories that were decayed
        """
        if now is None:
            now = time()
        
        # Get all memories
        memories = self.storage.get_all_memories()
        
        # Apply decay
        updated_memories, events = apply_decay(memories, now=now, limit=limit)
        
        # Save updated memories and events
        for memory in updated_memories:
            self.storage.update_memory(memory)
        
        for event in events:
            self.storage.save_event(event)
        
        return len(events)
    
    def get_events(self, memory_id: str) -> List[MemoryEvent]:
        """
        Get all events for a memory.
        
        Args:
            memory_id: Memory ID to get events for
            
        Returns:
            List of MemoryEvent
        """
        return self.storage.get_events(memory_id)
    
    def close(self) -> None:
        """
        Close the engine and release resources.
        """
        # SQLite connections are per-operation, so nothing to close here
        pass
