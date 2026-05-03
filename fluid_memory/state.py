"""
Fluid Memory State Constants and Bounds

Defines the state field bounds and default values for memory items.
"""

from typing import Final

# Value bounds for state fields
MIN_SALIENCE: Final[float] = 0.0
MAX_SALIENCE: Final[float] = 1.0
DEFAULT_SALIENCE: Final[float] = 0.5

MIN_CONFIDENCE: Final[float] = 0.0
MAX_CONFIDENCE: Final[float] = 1.0
DEFAULT_CONFIDENCE: Final[float] = 0.5

MIN_VOLATILITY: Final[float] = 0.0
MAX_VOLATILITY: Final[float] = 1.0
DEFAULT_VOLATILITY: Final[float] = 0.3

MIN_STABILITY: Final[float] = 0.0
MAX_STABILITY: Final[float] = 1.0
DEFAULT_STABILITY: Final[float] = 0.5

MIN_DECAY_RATE: Final[float] = 0.0
MAX_DECAY_RATE: Final[float] = 1.0
DEFAULT_DECAY_RATE: Final[float] = 0.05

MIN_LINK_STRENGTH: Final[float] = 0.0
MAX_LINK_STRENGTH: Final[float] = 1.0
DEFAULT_LINK_STRENGTH: Final[float] = 0.5

# Scoring boost amounts
DEFAULT_ACCESS_SALIENCE_BOOST: Final[float] = 0.02
DEFAULT_REINFORCEMENT_BOOST: Final[float] = 0.1
DEFAULT_CONTRADICTION_PENALTY: Final[float] = 0.1

# Decay constants
SECONDS_PER_DAY: Final[float] = 86400.0
DEFAULT_RECENCY_HALF_LIFE_DAYS: Final[float] = 1.0
MIN_ACTIVATION_FLOOR: Final[float] = 0.05


def clamp01(value: float) -> float:
    """
    Clamp a value to [0.0, 1.0] range.
    
    Args:
        value: The value to clamp
        
    Returns:
        Value bounded in [0.0, 1.0]
    """
    return max(0.0, min(1.0, value))
