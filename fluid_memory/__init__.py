"""
Fluid Memory Core - Standalone Adaptive Memory State Engine

A dynamic memory field where memories shift over time when touched by
new input, reuse, contradictions, confirmations, age, and importance.

Usage:
    from fluid_memory import FluidMemoryEngine, FluidMemoryConfig
    
    config = FluidMemoryConfig(data_dir="/path/to/data")
    engine = FluidMemoryEngine(config)
    
    # Add a memory
    memory = engine.add_memory("Important fact", tags=["key"])
    
    # Retrieve with fluid scoring
    results = engine.retrieve("important", limit=5)
    
    # Reinforce to strengthen
    engine.reinforce(memory.memory_id)
    
    # Apply decay to old memories
    engine.apply_decay()
    
    engine.close()
"""

from fluid_memory.config import FluidMemoryConfig
from fluid_memory.models import MemoryItem, RetrievalResult
from fluid_memory.events import MemoryEvent
from fluid_memory.engine import FluidMemoryEngine
from fluid_memory.exceptions import (
    FluidMemoryError,
    MemoryNotFoundError,
    DuplicateMemoryError,
    InvalidStateError,
)
from fluid_memory.events import EventType

__version__ = "1.0.0"

__all__ = [
    # Core classes
    "FluidMemoryEngine",
    "FluidMemoryConfig",
    "MemoryItem",
    "RetrievalResult",
    "MemoryEvent",
    "EventType",
    # Exceptions
    "FluidMemoryError",
    "MemoryNotFoundError",
    "DuplicateMemoryError",
    "InvalidStateError",
]
