"""
Fluid Memory Decay

Handles temporal decay of activation values.
Activation naturally decreases over time to simulate forgetting.
"""

from time import time
from typing import List, Optional
from m_flow.memory.fluid.models import FluidMemoryState


def compute_decayed_activation(
    current_activation: float,
    last_touched: float,
    decay_rate: float,
    now: Optional[float] = None,
) -> float:
    """
    Compute decayed activation based on time elapsed.
    
    Uses exponential decay: activation * exp(-decay_rate * time_elapsed)
    
    Args:
        current_activation: Current activation value [0-1]
        last_touched: Unix timestamp when node was last touched
        decay_rate: Decay rate constant (default 0.01 = ~1% per unit time)
        now: Current timestamp (defaults to time.time())
        
    Returns:
        Decayed activation value
    """
    if now is None:
        now = time()
    
    time_elapsed = now - last_touched
    
    # Exponential decay: activation decreases by factor of e^(-decay_rate * time)
    # This gives smooth, continuous decay that never quite reaches zero
    decayed = current_activation * (2.718281828 ** (-decay_rate * time_elapsed))
    
    return max(0.0, min(1.0, decayed))


def apply_decay(
    states: List[FluidMemoryState],
    now: Optional[float] = None,
    min_activation: float = 0.01,
) -> List[FluidMemoryState]:
    """
    Apply decay to all states in the list.
    
    Args:
        states: List of fluid states to decay
        now: Current timestamp (defaults to time.time())
        min_activation: Minimum activation floor (prevents true zero)
        
    Returns:
        The same list with decayed activation values
    """
    if now is None:
        now = time()
    
    for state in states:
        decayed = compute_decayed_activation(
            current_activation=state.activation,
            last_touched=state.last_touched_at,
            decay_rate=state.decay_rate,
            now=now,
        )
        # Apply floor to prevent true zero activation
        state.activation = max(min_activation, decayed)
    
    return states


def compute_recency_score(
    last_touched: float,
    now: Optional[float] = None,
    half_life: float = 86400.0,  # 24 hours in seconds
) -> float:
    """
    Compute a recency score based on time since last touch.
    
    Uses a half-life decay: score = 1.0 at touch time,
    0.5 after half_life seconds, approaching 0 over time.
    
    Args:
        last_touched: Unix timestamp of last touch
        now: Current timestamp (defaults to time.time())
        half_life: Time in seconds for score to decay to 0.5
        
    Returns:
        Recency score [0-1]
    """
    if now is None:
        now = time()
    
    time_elapsed = now - last_touched
    
    # Half-life decay: score = 0.5^(time_elapsed / half_life)
    score = 0.5 ** (time_elapsed / half_life)
    
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
        The same list with updated recency scores
    """
    if now is None:
        now = time()
    
    for state in states:
        state.recency_score = compute_recency_score(
            last_touched=state.last_touched_at,
            now=now,
        )
    
    return states
