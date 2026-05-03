"""
Fluid Memory Scoring

Two scoring paths:

1. compute_effective_score() — primary composite formula (v2):

       effective_score = (
           semantic_score  * 0.55 +
           graph_score     * 0.20 +
           activation_score * 0.15 +
           trust_score     * 0.10
       )

   Where:
       activation_score = state.activation * 0.6 + state.recency_score * 0.4
       trust_score      = state.source_trust * 0.7 + state.legal_weight * 0.3

   Contradiction pressure applies as a multiplicative penalty on the
   final score (not additive), capped at a 30% reduction.

   Higher score = better match (similarity convention).

2. fluid_score() / fluid_score_similarity() — DEPRECATED legacy path.
   Kept for backward compat; delegates to compute_effective_score internally.

Architecture principle — Activation ≠ Truth:
   Activation reflects recency of attention.
   Trust comes from provenance, cross-confirmation, and legal authority.
   The two signals are now separated and weighted independently.
"""

from typing import Any, Optional, Dict
from m_flow.memory.fluid.models import FluidMemoryState

# ---------------------------------------------------------------------------
# Legacy boost weights (kept for the fluid_score deprecated path)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Effective score component weights (must sum to 1.0)
# ---------------------------------------------------------------------------
_W_SEMANTIC    = 0.55
_W_GRAPH       = 0.20
_W_ACTIVATION  = 0.15
_W_TRUST       = 0.10

_MAX_CONTRADICTION_PENALTY = 0.30  # contradiction can reduce score by at most 30%


def _compute_activation_score(state: FluidMemoryState) -> float:
    """Internal: normalized activation signal [0, 1]."""
    return state.activation * 0.6 + state.recency_score * 0.4


def _compute_trust_score(state: FluidMemoryState) -> float:
    """Internal: normalized trust signal [0, 1]."""
    return state.source_trust * 0.7 + state.legal_weight * 0.3


def _apply_contradiction_penalty(score: float, contradiction_pressure: float) -> float:
    """
    Apply contradiction pressure as a multiplicative penalty.

    A node with high contradiction pressure is less reliable, so its
    score is reduced.  The reduction is capped at MAX_CONTRADICTION_PENALTY
    to prevent any node from being completely suppressed.

    Args:
        score: Current effective score before penalty
        contradiction_pressure: Contradiction pressure [0, 1]

    Returns:
        Penalised score
    """
    penalty_fraction = min(_MAX_CONTRADICTION_PENALTY, contradiction_pressure * _MAX_CONTRADICTION_PENALTY)
    return score * (1.0 - penalty_fraction)


def compute_effective_score(
    semantic_score: float,
    state: FluidMemoryState,
    graph_score: float = 0.0,
    w_semantic: float = _W_SEMANTIC,
    w_graph: float = _W_GRAPH,
    w_activation: float = _W_ACTIVATION,
    w_trust: float = _W_TRUST,
) -> float:
    """
    Compute the composite effective retrieval score (v2, primary path).

    Formula:
        effective_score = (
            semantic_score   * w_semantic  +   # vector similarity (dominant)
            graph_score      * w_graph     +   # graph hop proximity
            activation_score * w_activation +  # attention / recency
            trust_score      * w_trust         # provenance + legal authority
        )

    Contradiction pressure is then applied as a multiplicative penalty
    (capped at 30% reduction), enforcing the principle that contradicted
    nodes rank lower without being fully suppressed.

    Higher score = better match (similarity convention, opposite of distance).

    Args:
        semantic_score: Similarity score from vector retrieval [0, 1]
        state: Fluid memory state for the node
        graph_score: Graph proximity score [0, 1]; 0 if unavailable
        w_semantic: Weight for semantic component (default 0.55)
        w_graph: Weight for graph component (default 0.20)
        w_activation: Weight for activation component (default 0.15)
        w_trust: Weight for trust component (default 0.10)

    Returns:
        Composite effective score [0, 1] after contradiction penalty
    """
    activation_score = _compute_activation_score(state)
    trust_score = _compute_trust_score(state)

    raw = (
        semantic_score  * w_semantic +
        graph_score     * w_graph +
        activation_score * w_activation +
        trust_score     * w_trust
    )

    # Normalise raw to [0, 1] (weights sum to 1.0, inputs bounded to [0,1])
    effective = max(0.0, min(1.0, raw))

    # Apply contradiction penalty (multiplicative, not additive)
    return _apply_contradiction_penalty(effective, state.contradiction_pressure)


def explain_effective_score(
    semantic_score: float,
    state: FluidMemoryState,
    graph_score: float = 0.0,
    w_semantic: float = _W_SEMANTIC,
    w_graph: float = _W_GRAPH,
    w_activation: float = _W_ACTIVATION,
    w_trust: float = _W_TRUST,
) -> Dict:
    """
    Return a detailed breakdown of the effective score computation.

    Returns a dict with:
        semantic_score      — raw semantic input
        graph_score         — raw graph input
        activation_score    — normalized activation+recency signal
        trust_score         — normalized source_trust+legal_weight signal
        raw_score           — weighted sum before penalty
        contradiction_penalty_fraction — fraction of score removed
        final_score         — score after contradiction penalty
        components          — per-weight contributions
    """
    activation_score = _compute_activation_score(state)
    trust_score = _compute_trust_score(state)

    components = {
        "semantic":    semantic_score  * w_semantic,
        "graph":       graph_score     * w_graph,
        "activation":  activation_score * w_activation,
        "trust":       trust_score     * w_trust,
    }
    raw = sum(components.values())
    effective = max(0.0, min(1.0, raw))
    penalty_fraction = min(_MAX_CONTRADICTION_PENALTY, state.contradiction_pressure * _MAX_CONTRADICTION_PENALTY)
    final = effective * (1.0 - penalty_fraction)

    return {
        "semantic_score": semantic_score,
        "graph_score": graph_score,
        "activation_score": activation_score,
        "trust_score": trust_score,
        "raw_score": raw,
        "contradiction_pressure": state.contradiction_pressure,
        "contradiction_penalty_fraction": penalty_fraction,
        "final_score": final,
        "components": components,
        "weights": {
            "semantic": w_semantic,
            "graph": w_graph,
            "activation": w_activation,
            "trust": w_trust,
        },
    }


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
    DEPRECATED — use compute_effective_score() for new code.

    Legacy distance-based score adjustment (lower = better).
    Kept for backward compat with bundle_search.py callers that pass
    a distance score and expect a smaller value back when boosted.

    Args:
        base_retrieval_score: Base distance score (lower is better)
        state: Fluid memory state for the node
        max_impact: Absolute cap on adjustment (default 0.15)
        max_fraction: Relative cap as fraction of base score (default 0.30)
        custom_weights: Optional weight overrides

    Returns:
        Adjusted distance score (lower is better)
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
    DEPRECATED — use compute_effective_score() for new code.

    Legacy similarity-based score adjustment (higher = better).

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
) -> Dict[str, Any]:
    """
    DEPRECATED — use explain_effective_score() for new code.

    Return a detailed breakdown of the legacy fluid score adjustment.

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
