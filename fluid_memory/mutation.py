"""
Fluid Memory Mutation

Controlled mutation of memory content and state.
"""

import hashlib
from typing import Optional, Dict, Any
from time import time

from fluid_memory.models import MemoryItem
from fluid_memory.state import clamp01
from fluid_memory.events import MemoryEvent, EventType


def compute_mutation_resistance(
    stability: float,
    volatility: float,
    resistance_enabled: bool = True,
) -> float:
    """
    Compute how much a memory resists mutation.
    
    Higher stability = more resistance.
    Higher volatility = less resistance.
    
    Args:
        stability: Memory stability
        volatility: Memory volatility
        resistance_enabled: Whether resistance is active
        
    Returns:
        Resistance factor in [0.0, 1.0]
    """
    if not resistance_enabled:
        return 0.0
    
    # Stability increases resistance, volatility decreases it
    resistance = stability * (1.0 - volatility)
    return clamp01(resistance)


def apply_state_delta(
    memory: MemoryItem,
    state_delta: Dict[str, Any],
    resistance: float = 0.0,
) -> Dict[str, Any]:
    """
    Apply a state delta to memory, respecting resistance.
    
    Args:
        memory: Memory to mutate
        state_delta: Dict of field changes
        resistance: Mutation resistance factor
        
    Returns:
        Applied changes (may be reduced due to resistance)
    """
    applied = {}
    
    for field, new_value in state_delta.items():
        if not hasattr(memory, field):
            continue
        
        old_value = getattr(memory, field)
        
        # For float fields, apply resistance to the change
        if isinstance(new_value, float) and isinstance(old_value, float):
            change = new_value - old_value
            resisted_change = change * (1.0 - resistance)
            final_value = old_value + resisted_change
            setattr(memory, field, clamp01(final_value))
            applied[field] = {"old": old_value, "new": clamp01(final_value)}
        # For int fields
        elif isinstance(new_value, int) and isinstance(old_value, int):
            change = new_value - old_value
            resisted_change = int(change * (1.0 - resistance))
            final_value = old_value + resisted_change
            setattr(memory, field, final_value)
            applied[field] = {"old": old_value, "new": final_value}
        # For list fields (append only)
        elif isinstance(new_value, list) and isinstance(old_value, list):
            # Add new items
            existing_set = set(old_value)
            new_items = [item for item in new_value if item not in existing_set]
            if new_items:
                setattr(memory, field, old_value + new_items)
                applied[field] = {"added": new_items}
        # For other fields, replace if resistance is low
        else:
            if resistance < 0.5:  # Only replace if resistance < 50%
                setattr(memory, field, new_value)
                applied[field] = {"old": old_value, "new": new_value}
    
    return applied


def mutate_memory(
    memory: MemoryItem,
    new_content: Optional[str] = None,
    state_delta: Optional[Dict[str, Any]] = None,
    reason: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    mutation_resistance_enabled: bool = True,
) -> tuple[MemoryItem, MemoryEvent]:
    """
    Perform controlled mutation on a memory.
    
    Args:
        memory: Memory to mutate
        new_content: New content (if changing content)
        state_delta: State field changes
        reason: Reason for mutation
        metadata: Additional metadata
        mutation_resistance_enabled: Whether to apply resistance
        
    Returns:
        Tuple of (updated memory, mutation event)
    """
    now = time()
    old_hash = memory.content_hash
    applied_changes = {}
    
    # Apply content change if provided
    if new_content is not None and new_content != memory.content:
        memory.content = new_content
        memory.content_hash = hashlib.sha256(new_content.encode("utf-8")).hexdigest()
        applied_changes["content"] = {
            "old_hash": old_hash,
            "new_hash": memory.content_hash,
        }
    
    # Apply state delta
    if state_delta:
        resistance = compute_mutation_resistance(
            memory.stability,
            memory.volatility,
            mutation_resistance_enabled,
        )
        state_changes = apply_state_delta(memory, state_delta, resistance)
        if state_changes:
            applied_changes["state"] = state_changes
    
    # Update timestamp
    memory.touch()
    
    # Create event
    event = MemoryEvent(
        event_id=str(hash(f"{memory.memory_id}_mutate_{now}")),
        memory_id=memory.memory_id,
        event_type=EventType.MUTATED,
        timestamp=now,
        delta_json=applied_changes,
        metadata_json={
            "reason": reason,
            **(metadata or {}),
        },
    )
    
    return memory, event
