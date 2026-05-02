"""Fluid Memory Decay Logic

Decay is lane-specific:
- time: General salience decay over time
- legal: Legal authority decay (slow)
- trust: Trustworthiness decay
- interest: Interestingness decay (fast)
- attention: Attention decay (very fast)
"""

from time import time
from typing import List, Optional, Tuple
from fluid_memory.models import MemoryItem, DecayEvent
from fluid_memory.config import FluidMemoryConfig
from fluid_memory.events import MemoryEvent, EventType

import uuid


# How many seconds old an updated_at timestamp must be before decay kicks in
_MIN_AGE_SECONDS = 3600.0  # 1 hour


def compute_decay_amount(
    salience: float,
    elapsed_days: float,
    decay_rate: float,
    stability: float,
) -> float:
    """Compute the amount to subtract from a salience-like value.

    Args:
        salience: Current value (not used in formula, kept for API symmetry)
        elapsed_days: Days elapsed since last update
        decay_rate: Per-day decay rate
        stability: Memory stability (0.0–1.0); higher = less decay

    Returns:
        Amount to decay (non-negative float)
    """
    effective_rate = decay_rate * (1.0 - stability)
    return effective_rate * elapsed_days


def apply_decay_to_memory(
    memory: MemoryItem,
    min_salience: float = 0.0,
) -> Tuple[MemoryItem, bool]:
    """Apply time-based salience decay to a single memory in-place.

    Only decays memories whose ``updated_at`` is at least one hour old.

    Args:
        memory: Memory item to potentially decay
        min_salience: Floor for salience after decay

    Returns:
        Tuple of (memory, was_decayed)
    """
    now = time()
    elapsed_seconds = now - memory.updated_at
    elapsed_days = elapsed_seconds / 86400.0

    if elapsed_seconds < _MIN_AGE_SECONDS:
        return memory, False

    decay_amount = compute_decay_amount(
        memory.salience,
        elapsed_days,
        memory.decay_rate,
        memory.stability,
    )
    new_salience = max(min_salience, memory.salience - decay_amount)
    memory.salience = new_salience
    memory.updated_at = now
    return memory, True


def apply_decay(
    memories: List[MemoryItem],
    limit: Optional[int] = None,
    min_salience: float = 0.0,
) -> Tuple[List[MemoryItem], List[MemoryEvent]]:
    """Apply decay to a collection of memories.

    Args:
        memories: List of memories to process
        limit: Maximum number to process (None = all)
        min_salience: Floor for salience after decay

    Returns:
        Tuple of (processed_memories, decay_events)
    """
    batch = memories if limit is None else memories[:limit]
    events: List[MemoryEvent] = []
    updated: List[MemoryItem] = []

    for memory in batch:
        old_salience = memory.salience
        mem, was_decayed = apply_decay_to_memory(memory, min_salience)
        updated.append(mem)
        if was_decayed:
            event = MemoryEvent(
                event_id=f"decay_{uuid.uuid4().hex[:12]}",
                memory_id=memory.memory_id,
                event_type=EventType.DECAYED,
                delta_json={
                    "old_salience": old_salience,
                    "new_salience": mem.salience,
                    "lane": "time",
                },
            )
            events.append(event)

    return updated, events


class DecayManager:
    """Manages decay calculations for memory items."""

    def __init__(self, config: FluidMemoryConfig):
        self.config = config
        self.lane_rates = {
            "time": config.default_decay_rate,
            "legal": config.legal_decay_rate,
            "trust": config.trust_decay_rate,
            "interest": config.interest_decay_rate,
            "attention": config.attention_decay_rate,
        }

    def calculate_decay(
        self,
        memory: MemoryItem,
        lane: str = "time",
        days: float = 1.0
    ) -> Tuple[float, List[DecayEvent]]:
        """Calculate decay for a memory item."""
        rate = self.lane_rates.get(lane, self.config.default_decay_rate)
        effective_rate = rate * (1.0 - memory.stability)
        decay_amount = effective_rate * days

        lane_values = {
            "time": memory.salience,
            "legal": memory.legal_salience,
            "trust": memory.trust_salience,
            "interest": memory.interest_salience,
            "attention": memory.attention_salience,
        }

        current_value = lane_values.get(lane, memory.salience)
        new_value = max(0.0, current_value - decay_amount)

        event = DecayEvent(
            event_id=f"decay_{uuid.uuid4().hex[:12]}",
            memory_id=memory.memory_id,
            lane=lane,
            before_value=current_value,
            after_value=new_value,
            decay_amount=decay_amount,
        )

        return new_value, [event]

    def apply_decay(
        self,
        memory: MemoryItem,
        lane: str = "time",
        days: float = 1.0
    ) -> DecayEvent:
        """Apply decay to a memory item in-place."""
        new_value, events = self.calculate_decay(memory, lane, days)

        if lane == "time":
            memory.salience = new_value
        elif lane == "legal":
            memory.legal_salience = new_value
        elif lane == "trust":
            memory.trust_salience = new_value
        elif lane == "interest":
            memory.interest_salience = new_value
        elif lane == "attention":
            memory.attention_salience = new_value

        memory.updated_at = time()
        return events[0]

    def apply_all_decay(self, memory: MemoryItem, days: float = 1.0) -> List[DecayEvent]:
        """Apply decay to all lanes."""
        events = []
        for lane in self.lane_rates.keys():
            event = self.apply_decay(memory, lane, days)
            if event:
                events.append(event)
        return events
