"""
Fluid Memory Configuration

Configuration class with sensible defaults for the fluid memory system.
"""

from typing import Optional
from pathlib import Path


class FluidMemoryConfig:
    """
    Configuration for the Fluid Memory Core.
    
    All fields have sensible defaults so the system works out of the box.
    
    Attributes:
        data_dir: Directory for SQLite database
        sqlite_path: Full path to SQLite database (derived from data_dir if not set)
        default_salience: Initial salience for new memories
        default_confidence: Initial confidence for new memories
        default_volatility: Initial volatility for new memories
        default_stability: Initial stability for new memories
        default_decay_rate: Initial decay rate for new memories
        access_salience_boost: Salience increase per access
        reinforcement_boost: Amount reinforcement strengthens memory
        contradiction_penalty: Amount contradiction weakens memory
        mutation_resistance_enabled: Whether stability resists mutation
    """
    
    def __init__(
        self,
        data_dir: Optional[str] = None,
        sqlite_path: Optional[str] = None,
        default_salience: float = 0.5,
        default_confidence: float = 0.5,
        default_volatility: float = 0.3,
        default_stability: float = 0.5,
        default_decay_rate: float = 0.05,
        access_salience_boost: float = 0.02,
        reinforcement_boost: float = 0.1,
        contradiction_penalty: float = 0.1,
        mutation_resistance_enabled: bool = True,
    ):
        self.data_dir = data_dir or ".fluid_memory"
        self.sqlite_path = sqlite_path or str(Path(self.data_dir) / "fluid_memory.db")
        self.default_salience = self._clamp01(default_salience)
        self.default_confidence = self._clamp01(default_confidence)
        self.default_volatility = self._clamp01(default_volatility)
        self.default_stability = self._clamp01(default_stability)
        self.default_decay_rate = self._clamp01(default_decay_rate)
        self.access_salience_boost = self._clamp01(access_salience_boost)
        self.reinforcement_boost = self._clamp01(reinforcement_boost)
        self.contradiction_penalty = self._clamp01(contradiction_penalty)
        self.mutation_resistance_enabled = mutation_resistance_enabled
    
    @staticmethod
    def _clamp01(value: float) -> float:
        """Clamp value to [0.0, 1.0]."""
        return max(0.0, min(1.0, value))
