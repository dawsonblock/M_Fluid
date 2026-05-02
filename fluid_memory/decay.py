"""Fluid Memory Decay Logic

Decay is lane-specific:
- time: General salience decay over time
- legal: Legal authority decay (slow)
- trust: Trustworthiness decay
- interest: Interestingness decay (fast)
- attention: Attention decay (very fast)
"""

from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from fluid_memory.models import MemoryItem, DecayEvent
from fluid_memory.config import FluidMemoryConfig

import uuid


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
        """Calculate decay for a memory item.

        Args:
            memory: The memory to decay
            lane: Decay lane (time, legal, trust, interest, attention)
            days: Number of days to simulate

        Returns:
            Tuple of (new_salience, list of decay events)
        """
        rate = self.lane_rates.get(lane, self.config.default_decay_rate)

        # Apply stability modifier (more stable = less decay)
        effective_rate = rate * (1.0 - memory.stability)

        # Calculate decay amount
        decay_amount = effective_rate * days

        # Get current lane value
        lane_values = {
            "time": memory.salience,
            "legal": memory.legal_salience,
            "trust": memory.trust_salience,
            "interest": memory.interest_salience,
            "attention": memory.attention_salience,
        }

        current_value = lane_values.get(lane, memory.salience)
        new_value = max(0.0, current_value - decay_amount)

        # Create decay event
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
        """Apply decay to a memory item in-place.

        Args:
            memory: The memory to modify
            lane: Decay lane
            days: Days to decay

        Returns:
            The decay event recorded
        """
        new_value, events = self.calculate_decay(memory, lane, days)

        # Update the appropriate lane
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

        memory.updated_at = datetime.utcnow()

        return events[0] if events else None

    def apply_all_decay(self, memory: MemoryItem, days: float = 1.0) -> List[DecayEvent]:
        """Apply decay to all lanes.

        Args:
            memory: The memory to modify
            days: Days to decay

        Returns:
            List of decay events
        """
        events = []
        for lane in self.lane_rates.keys():
            event = self.apply_decay(memory, lane, days)
            if event:
                events.append(event)
        return events
