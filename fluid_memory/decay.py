"""
Fluid Memory Decay

Time-based salience decay for unused memories.
"""

from typing import Optional, List
from time import time

from fluid_memory.models import MemoryItem
from fluid_memory.state import clamp01, SECONDS_PER_DAY
from fluid_memory.events import MemoryEvent, EventType


def compute_decay_amount(
    salience: float,
    elapsed_days: float,
    decay_rate: float,
    stability: float,
) -> float:
    """
    Compute the amount of decay to apply.
    
    Formula: elapsed_days * decay_rate * (1.0 - stability)
    
    Args:
        salience: Current salience value
        elapsed_days: Days since last access or update
        decay_rate: Decay rate per day
        stability: Stability (higher = less decay)
        
    Returns:
        Amount to reduce salience by
    """
    # Stable memories decay slower
    effective_rate = decay_rate * (1.0 - stability)
    decay_amount = elapsed_days * effective_rate
    return decay_amount


def apply_decay_to_memory(
    memory: MemoryItem,
    now: Optional[float] = None,
    min_salience: float = 0.0,
) -> tuple[MemoryItem, bool]:
    """
    Apply decay to a single memory.
    
    Args:
        memory: Memory to decay
        now: Current timestamp (defaults to time.time())
        min_salience: Floor for salience (never decay below this)
        
    Returns:
        Tuple of (updated memory, whether decay was applied)
    """
    if now is None:
        now = time()
    
    # Determine reference time (last access or last update)
    reference_time = memory.last_accessed_at or memory.updated_at
    elapsed_seconds = now - reference_time
    elapsed_days = elapsed_seconds / SECONDS_PER_DAY
    
    if elapsed_days <= 0:
        return memory, False
    
    # Compute decay
    decay_amount = compute_decay_amount(
        memory.salience,
        elapsed_days,
        memory.decay_rate,
        memory.stability,
    )
    
    old_salience = memory.salience
    new_salience = max(min_salience, memory.salience - decay_amount)
    
    if new_salience < old_salience:
        memory.salience = clamp01(new_salience)
        memory.update_timestamp()
        return memory, True
    
    return memory, False


def apply_decay(
    memories: List[MemoryItem],
    now: Optional[float] = None,
    limit: Optional[int] = None,
    min_salience: float = 0.0,
) -> tuple[List[MemoryItem], List[MemoryEvent]]:
    """
    Apply decay to a list of memories.
    
    Args:
        memories: List of memories to decay
        now: Current timestamp (defaults to time.time())
        limit: Maximum number of memories to process
        min_salience: Floor for salience
        
    Returns:
        Tuple of (updated memories, list of decay events)
    """
    if now is None:
        now = time()
    
    updated_memories = []
    events = []
    
    to_process = memories[:limit] if limit else memories
    
    for memory in to_process:
        updated_memory, was_decayed = apply_decay_to_memory(memory, now, min_salience)
        updated_memories.append(updated_memory)
        
        if was_decayed:
            event = MemoryEvent(
                event_id=str(hash(f"{memory.memory_id}_decay_{now}")),
                memory_id=memory.memory_id,
                event_type=EventType.DECAYED,
                timestamp=now,
                delta_json={"old_salience": memory.salience, "new_salience": updated_memory.salience},
            )
            events.append(event)
    
    return updated_memories, events
