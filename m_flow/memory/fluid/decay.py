"""
Fluid Memory Decay

Five-lane temporal decay model.  Each lane governs a specific signal dimension:

    ATTENTION_DECAY      = 0.20 / day   # short-term visibility (breaking news)
    INTEREST_DECAY       = 0.05 / day   # user/session engagement patterns
    TRUST_DECAY          = 0.000 / day  # provenance trust — NEVER decays
    LEGAL_DECAY          = 0.000 / day  # court/gov evidence — IMMUTABLE by policy
    CONTRADICTION_DECAY  = 0.01 / day   # conflict pressure eases slowly over time

Key design principle:
    Activation ≠ Truth.

    Activation decays quickly (attention / interest lanes).
    Trust and legal_weight are immutable — they derive from provenance, not recency.
    Contradiction pressure decreases when no new conflicts arrive.

Minimum activation floor: 0.05 (nodes never go fully dark).

Backward-compat aliases:
    SHORT_TERM_DECAY → ATTENTION_DECAY (0.20)
    NORMAL_DECAY     → INTEREST_DECAY  (0.05)
    LEGAL_DECAY      → 0.000 (was 0.002; now immutable)
"""

import math
from time import time
from typing import List, Optional
from m_flow.memory.fluid.models import FluidMemoryState

_SECONDS_PER_DAY = 86400.0

# ---------------------------------------------------------------------------
# 5-lane rate constants
# ---------------------------------------------------------------------------
ATTENTION_DECAY     = 0.20   # short-term: social amplification, breaking news
INTEREST_DECAY      = 0.05   # medium-term: user session patterns, ongoing interest
TRUST_DECAY         = 0.000  # trust is immutable — derived from provenance chain
LEGAL_DECAY         = 0.000  # legal evidence is immutable — court/government records
CONTRADICTION_DECAY = 0.01   # contradiction pressure eases when conflicts stop arriving

# ---------------------------------------------------------------------------
# Backward-compat aliases (keep callers from breaking)
# ---------------------------------------------------------------------------
SHORT_TERM_DECAY = ATTENTION_DECAY   # was 0.25, now 0.20
NORMAL_DECAY     = INTEREST_DECAY    # was 0.02, now 0.05
# LEGAL_DECAY left as 0.000 — previously 0.002, now immutable

_LANE_RATES: dict = {
    "attention":     ATTENTION_DECAY,
    "interest":      INTEREST_DECAY,
    "trust":         TRUST_DECAY,
    "legal":         LEGAL_DECAY,
    "contradiction": CONTRADICTION_DECAY,
    # Legacy lane names
    "short_term":    ATTENTION_DECAY,
    "normal":        INTEREST_DECAY,
}

DEFAULT_MIN_ACTIVATION = 0.05
DEFAULT_RECENCY_HALF_LIFE_DAYS = 1.0  # recency_score halves every 24 h


def _effective_activation_decay_rate(state: FluidMemoryState) -> float:
    """
    Determine the per-day decay rate for activation based on decay_lane.

    trust/legal lanes return 0.0 so activation on those nodes is effectively
    immutable when set explicitly, but normal episodic nodes use attention/interest.
    """
    return _LANE_RATES.get(state.decay_lane, state.decay_rate)


# Kept for backward compat
def _effective_decay_rate(state: FluidMemoryState) -> float:
    return _effective_activation_decay_rate(state)


def compute_decayed_activation(
    current_activation: float,
    last_touched: float,
    decay_rate: float,
    now: Optional[float] = None,
    min_activation: float = DEFAULT_MIN_ACTIVATION,
) -> float:
    """
    Compute exponentially-decayed activation.

    Formula: activation * exp(-decay_rate * elapsed_days)
    Floor:   max(min_activation, result)

    Args:
        current_activation: Current activation value [0, 1]
        last_touched: Unix timestamp of last touch
        decay_rate: Per-day rate constant (0.0 = immutable)
        now: Current Unix timestamp (defaults to time.time())
        min_activation: Floor — nodes never reach zero (default 0.05)

    Returns:
        Decayed activation in [min_activation, 1.0]
    """
    if now is None:
        now = time()

    if decay_rate == 0.0:
        return max(min_activation, min(1.0, current_activation))

    elapsed_days = (now - last_touched) / _SECONDS_PER_DAY
    decayed = current_activation * math.exp(-decay_rate * elapsed_days)
    return max(min_activation, min(1.0, decayed))


def compute_decayed_contradiction_pressure(
    current_pressure: float,
    last_touched: float,
    now: Optional[float] = None,
    decay_rate: float = CONTRADICTION_DECAY,
) -> float:
    """
    Ease contradiction pressure over time.

    Contradiction pressure decays toward 0.0 (no floor) when no new
    conflicts arrive, representing resolution or fading relevance.

    Args:
        current_pressure: Current contradiction pressure [0, 1]
        last_touched: Unix timestamp of when pressure was last updated
        now: Current Unix timestamp
        decay_rate: Per-day easing rate (default CONTRADICTION_DECAY = 0.01)

    Returns:
        Eased pressure in [0.0, 1.0]
    """
    if now is None:
        now = time()

    elapsed_days = (now - last_touched) / _SECONDS_PER_DAY
    eased = current_pressure * math.exp(-decay_rate * elapsed_days)
    return max(0.0, min(1.0, eased))


def apply_decay(
    states: List[FluidMemoryState],
    now: Optional[float] = None,
    min_activation: float = DEFAULT_MIN_ACTIVATION,
) -> List[FluidMemoryState]:
    """
    Apply per-day decay to all states.

    What decays:
        - activation        → attention_decay or interest_decay (by lane)
        - contradiction_pressure → contradiction_decay (eases toward 0)

    What does NOT decay:
        - source_trust      → TRUST_DECAY = 0.0 (provenance is immutable)
        - legal_weight      → LEGAL_DECAY = 0.0 (court evidence is immutable)
        - confidence        → reflects evidence quality, not recency
        - recency_score     → updated separately via update_recency_scores()

    Args:
        states: List of fluid states to decay (mutated in-place)
        now: Current timestamp (defaults to time.time())
        min_activation: Floor to prevent nodes reaching zero

    Returns:
        The same list with updated fields
    """
    if now is None:
        now = time()

    for state in states:
        # Activation decay (lane-aware)
        act_rate = _effective_activation_decay_rate(state)
        state.activation = compute_decayed_activation(
            current_activation=state.activation,
            last_touched=state.last_touched_at,
            decay_rate=act_rate,
            now=now,
            min_activation=min_activation,
        )

        # Contradiction pressure easing
        state.contradiction_pressure = compute_decayed_contradiction_pressure(
            current_pressure=state.contradiction_pressure,
            last_touched=state.last_touched_at,
            now=now,
        )

        # Trust and legal_weight intentionally NOT touched:
        # state.source_trust unchanged
        # state.legal_weight unchanged

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
        half_life_days: Days until score reaches 0.5 (default = 1 day)

    Returns:
        Recency score [0, 1]
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
        states: List of fluid states to update (mutated in-place)
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
