"""
Fluid Memory Event System

Event types and structures for tracking all memory state changes.
"""

from enum import Enum
from typing import Optional, Any
from pydantic import BaseModel, Field
from time import time


class EventType(str, Enum):
    """Types of memory events."""
    CREATED = "created"
    ACCESSED = "accessed"
    REINFORCED = "reinforced"
    CONTRADICTED = "contradicted"
    DECAYED = "decayed"
    MUTATED = "mutated"
    LINKED = "linked"
    DELETED = "deleted"


class MemoryEvent(BaseModel):
    """
    A recorded event describing a change to memory state.
    
    Fields:
        event_id: Unique identifier for this event
        memory_id: ID of the affected memory
        event_type: Type of event (created, accessed, etc.)
        timestamp: Unix timestamp when event occurred
        delta_json: JSON-serializable dict of changes
        metadata_json: Additional event metadata
    """
    event_id: str
    memory_id: str
    event_type: EventType
    timestamp: float = Field(default_factory=time)
    delta_json: dict = Field(default_factory=dict)
    metadata_json: dict = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            # Handle any custom serialization needs
        }


def create_event(
    event_id: str,
    memory_id: str,
    event_type: EventType,
    delta: Optional[dict] = None,
    metadata: Optional[dict] = None,
) -> MemoryEvent:
    """
    Factory function to create a memory event.
    
    Args:
        event_id: Unique event identifier
        memory_id: Affected memory ID
        event_type: Type of event
        delta: Changes made (old/new values)
        metadata: Additional context
        
    Returns:
        MemoryEvent instance
    """
    return MemoryEvent(
        event_id=event_id,
        memory_id=memory_id,
        event_type=event_type,
        delta_json=delta or {},
        metadata_json=metadata or {},
    )
