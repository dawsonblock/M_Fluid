"""
Fluid Memory Scoring

Integration with retrieval scoring system.
Provides fluid_score() to boost/penalize retrieval scores.

Scoring is bounded to prevent the fluid layer from dominating:
  - Raw boost is computed from weighted state factors.
  - Boost is capped at min(max_boost_impact, base_score * max_boost_fraction).
  - Default caps: absolute 0.15, relative 30% of base score.

This means fluid memory cannot change a score by more than 30% or
0.15 absolute units, whichever is smaller.  The base retrieval signal
always remains dominant.
"""

from typing import Optional, Dict
from m_flow.memory.fluid.models import FluidMemoryState

_DEFAULT_WEIGHTS: Dict[str, float] = {
    "activation": 0.25,
    "confidence": 0.25,
    "source_trust": 0.15,
    "recency_score": 0.10,
    "salience": 0.10,
    "legal_weight": 0.10,
    "contradiction_pressure": 0.20,  # applied as negative
}

_DEFAULT_MAX_IMPACT = 0.15
_DEFAULT_MAX_FRACTION = 0.30


def compute_fluid_boost(
    state: FluidMemoryState,
    custom_weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Compute raw fluid boost (unbounded).

    Args:
        state: Fluid memory state
        custom_weights: Optional weight overrides

    Returns:
        Raw boost value (can be positive or negative, unbounded)
    """
    w = dict(_DEFAULT_WEIGHTS)
    if custom_weights:
        w.update(custom_weights)

    return (
        state.activation * w["activation"]
        + state.confidence * w["confidence"]
        + state.source_trust * w["source_trust"]
        + state.recency_score * w["recency_score"]
        + state.salience * w["salience"]
        + state.legal_weight * w["legal_weight"]
        - state.contradiction_pressure * w["contradiction_pressure"]
    )


def _bound_boost(
    raw_boost: float,
    base_score: float,
    max_impact: float = _DEFAULT_MAX_IMPACT,
    max_fraction: float = _DEFAULT_MAX_FRACTION,
) -> float:
    """
    Apply bounds to a raw boost value.

    Cap at min(max_impact, |base_score| * max_fraction),
    preserving the sign of the boost.
    """
    cap = min(max_impact, abs(base_score) * max_fraction)
    return max(-cap, min(cap, raw_boost))


def fluid_score(
    base_retrieval_score: float,
    state: FluidMemoryState,
    max_impact: float = _DEFAULT_MAX_IMPACT,
    max_fraction: float = _DEFAULT_MAX_FRACTION,
    custom_weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Compute fluid-adjusted retrieval score (distance-based, lower = better).

    The boost is bounded so fluid memory never dominates base retrieval.

    Args:
        base_retrieval_score: Base distance score (lower is better)
        state: Fluid memory state for the node
        max_impact: Absolute cap on adjustment (default 0.15)
        max_fraction: Relative cap as fraction of base score (default 0.30)
        custom_weights: Optional weight overrides

    Returns:
        Adjusted score (lower is better)
    """
    raw = compute_fluid_boost(state, custom_weights)
    bounded = _bound_boost(raw, base_retrieval_score, max_impact, max_fraction)
    return base_retrieval_score - bounded


def fluid_score_similarity(
    base_similarity_score: float,
    state: FluidMemoryState,
    max_impact: float = _DEFAULT_MAX_IMPACT,
    max_fraction: float = _DEFAULT_MAX_FRACTION,
    custom_weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Compute fluid-adjusted similarity score (higher = better).

    Args:
        base_similarity_score: Base similarity score (higher is better)
        state: Fluid memory state for the node
        max_impact: Absolute cap on adjustment
        max_fraction: Relative cap as fraction of base score

    Returns:
        Adjusted similarity score (higher is better)
    """
    raw = compute_fluid_boost(state, custom_weights)
    bounded = _bound_boost(raw, base_similarity_score, max_impact, max_fraction)
    return base_similarity_score + bounded


def explain_fluid_score(
    base_retrieval_score: float,
    state: FluidMemoryState,
    max_impact: float = _DEFAULT_MAX_IMPACT,
    max_fraction: float = _DEFAULT_MAX_FRACTION,
) -> Dict[str, float]:
    """
    Return a detailed breakdown of the fluid score adjustment.

    Useful for debugging, logging, and evaluation.

    Returns a dict with:
        base_score          — original score before fluid adjustment
        final_score         — score after bounded boost applied
        raw_boost           — unbounded boost from state factors
        bounded_boost       — clamped boost actually applied
        cap_applied         — cap value used
        components          — individual factor contributions
    """
    w = _DEFAULT_WEIGHTS
    components = {
        "activation":            state.activation * w["activation"],
        "confidence":            state.confidence * w["confidence"],
        "source_trust":          state.source_trust * w["source_trust"],
        "recency_score":         state.recency_score * w["recency_score"],
        "salience":              state.salience * w["salience"],
        "legal_weight":          state.legal_weight * w["legal_weight"],
        "contradiction_penalty": -(state.contradiction_pressure * w["contradiction_pressure"]),
    }
    raw = sum(components.values())
    cap = min(max_impact, abs(base_retrieval_score) * max_fraction)
    bounded = max(-cap, min(cap, raw))
    final = base_retrieval_score - bounded

    return {
        "base_score": base_retrieval_score,
        "final_score": final,
        "raw_boost": raw,
        "bounded_boost": bounded,
        "cap_applied": cap,
        "components": components,
    }


def should_boost_retrieval(
    state: FluidMemoryState,
    min_activation: float = 0.1,
    min_confidence: float = 0.3,
) -> bool:
    """
    Quick eligibility check before applying fluid score.

    Returns False for heavily-contradicted or effectively-inactive nodes
    so the scorer can skip them entirely for performance.
    """
    if state.contradiction_pressure > 0.5:
        return False
    if state.activation < min_activation and state.confidence < min_confidence:
        return False
    return True
