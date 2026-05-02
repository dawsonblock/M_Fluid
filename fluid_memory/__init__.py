"""Fluid Memory - Standalone adaptive memory engine.

A dynamic memory system where memories shift over time based on
activation, decay, confidence, salience, and contradictions.

This is a standalone package - no M-flow dependencies required.
"""

from fluid_memory.config import FluidMemoryConfig
from fluid_memory.models import MemoryItem, MemoryLink, RetrievalResult
from fluid_memory.events import MemoryEvent, EventType
from fluid_memory.engine import FluidMemoryEngine
from fluid_memory.exceptions import (
    FluidMemoryError,
    StorageError,
    MemoryNotFoundError,
    DuplicateMemoryError,
)

__version__ = "0.1.0"

__all__ = [
    "FluidMemoryConfig",
    "FluidMemoryEngine",
    "MemoryItem",
    "MemoryLink",
    "RetrievalResult",
    "MemoryEvent",
    "EventType",
    "FluidMemoryError",
    "StorageError",
    "MemoryNotFoundError",
    "DuplicateMemoryError",
]
