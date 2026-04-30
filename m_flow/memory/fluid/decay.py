"""
Fluid Memory Decay

Handles temporal decay of activation values.
Activation naturally decreases over time to simulate forgetting.

Time unit: DAYS (not seconds).
Decay affects only activation and recency_score.
Source trust, confidence, and legal_weight never decay.

Decay lanes (per-day rates):
    SHORT_TERM_DECAY = 0.25   # temporary attention — loses ~22% per day
    NORMAL_DECAY     = 0.02   # standard memory     — loses ~2% per day
    LEGAL_DECAY      = 0.002  # legal evidence      — loses ~0.2% per day

Minimum activation floor: 0.05 (nodes never go fully dark).
"""

import math
from time import time
from typing import List, Optional
from m_flow.memory.fluid.models import FluidMemoryState

_SECONDS_PER_DAY = 86400.0

SHORT_TERM_DECAY = 0.25
NORMAL_DECAY = 0.02
LEGAL_DECAY = 0.002

_LANE_RATES = {
    "short_term": SHORT_TERM_DECAY,
    "normal": NORMAL_DECAY,
    "legal": LEGAL_DECAY,
}

DEFAULT_MIN_ACTIVATION = 0.05
DEFAULT_RECENCY_HALF_LIFE_DAYS = 1.0   # recency_score halves every 24 h


def _effective_decay_rate(state: FluidMemoryState) -> float:
    """Return the per-day decay rate, preferring the lane over the raw field."""
    lane_rate = _LANE_RATES.get(state.decay_lane)
    if lane_rate is not None:
        return lane_rate
    return state.decay_rate


def compute_decayed_activation(
    current_activation: float,
    last_touched: float,
    decay_rate: float,
    now: Optional[float] = None,
    min_activation: float = DEFAULT_MIN_ACTIVATION,
) -> float:
    """
    Compute decayed activation based on time elapsed.

    Uses exponential decay: activation * exp(-decay_rate * elapsed_days)

    Args:
        current_activation: Current activation value [0-1]
        last_touched: Unix timestamp when node was last touched
        decay_rate: Per-day decay rate constant
        now: Current Unix timestamp (defaults to time.time())
        min_activation: Floor value (default 0.05)

    Returns:
        Decayed activation value, clamped to [min_activation, 1.0]
    """
    if now is None:
        now = time()

    elapsed_days = (now - last_touched) / _SECONDS_PER_DAY
    decayed = current_activation * math.exp(-decay_rate * elapsed_days)
    return max(min_activation, min(1.0, decayed))


def apply_decay(
    states: List[FluidMemoryState],
    now: Optional[float] = None,
    min_activation: float = DEFAULT_MIN_ACTIVATION,
) -> List[FluidMemoryState]:
    """
    Apply per-day decay to all states in the list.

    Only activation is decayed. source_trust, confidence, and
    legal_weight are left untouched — they reflect evidence quality,
    not recency of attention.

    Args:
        states: List of fluid states to decay
        now: Current timestamp (defaults to time.time())
        min_activation: Floor to prevent nodes reaching zero

    Returns:
        The same list with updated activation values (mutated in-place)
    """
    if now is None:
        now = time()

    for state in states:
        rate = _effective_decay_rate(state)
        state.activation = compute_decayed_activation(
            current_activation=state.activation,
            last_touched=state.last_touched_at,
            decay_rate=rate,
            now=now,
            min_activation=min_activation,
        )

    return states


def compute_recency_score(
    last_touched: float,
    now: Optional[float] = None,
    half_life_days: float = DEFAULT_RECENCY_HALF_LIFE_DAYS,
) -> float:
    """
    Compute a recency score based on time since last touch.

    Uses half-life decay: 1.0 immediately after touch,
    0.5 after half_life_days, approaching 0 over time.

    Args:
        last_touched: Unix timestamp of last touch
        now: Current timestamp (defaults to time.time())
        half_life_days: Days for score to reach 0.5 (default = 1 day)

    Returns:
        Recency score [0-1]
    """
    if now is None:
        now = time()

    elapsed_days = (now - last_touched) / _SECONDS_PER_DAY
    score = 0.5 ** (elapsed_days / half_life_days)
    return max(0.0, min(1.0, score))


def update_recency_scores(
    states: List[FluidMemoryState],
    now: Optional[float] = None,
) -> List[FluidMemoryState]:
    """
    Update recency scores for all states.

    Args:
        states: List of fluid states to update
        now: Current timestamp (defaults to time.time())

    Returns:
        The same list with updated recency_score values
    """
    if now is None:
        now = time()

    for state in states:
        state.recency_score = compute_recency_score(
            last_touched=state.last_touched_at,
            now=now,
        )

    return states
