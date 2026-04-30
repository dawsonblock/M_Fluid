"""
Fluid Memory Scoring

Integration with retrieval scoring system.
Provides fluid_score() function to boost/penalize retrieval scores.
"""

from typing import Optional
from m_flow.memory.fluid.models import FluidMemoryState


def fluid_score(
    base_retrieval_score: float, 
    state: FluidMemoryState,
) -> float:
    """
    Compute fluid-boosted retrieval score.
    
    Combines base retrieval score (from vector similarity) with
    fluid state factors (activation, confidence, trust, etc.).
    
    Scoring weights:
    - activation: 0.25 (currently active nodes get boost)
    - confidence: 0.25 (high confidence nodes get boost)
    - source_trust: 0.15 (trusted sources get boost)
    - recency_score: 0.10 (recent nodes get boost)
    - salience: 0.10 (salient nodes get boost)
    - legal_weight: 0.10 (legal docs get boost)
    - contradiction_pressure: -0.20 (contradicted nodes get penalty)
    
    Args:
        base_retrieval_score: Base score from vector similarity (lower is better)
        state: Fluid memory state for the node
        
    Returns:
        Adjusted score (lower is better)
    """
    boost = (
        state.activation * 0.25
        + state.confidence * 0.25
        + state.source_trust * 0.15
        + state.recency_score * 0.10
        + state.salience * 0.10
        + state.legal_weight * 0.10
        - state.contradiction_pressure * 0.20
    )
    
    # Apply boost to base score
    # For distance-based scores (lower=better), subtract boost
    # For similarity scores (higher=better), add boost
    return base_retrieval_score - boost


def fluid_score_similarity(
    base_similarity_score: float,
    state: FluidMemoryState,
) -> float:
    """
    Compute fluid-boosted similarity score.
    
    Version for similarity-based scoring where higher is better.
    
    Args:
        base_similarity_score: Base similarity score (higher is better)
        state: Fluid memory state for the node
        
    Returns:
        Adjusted similarity score (higher is better)
    """
    boost = (
        state.activation * 0.25
        + state.confidence * 0.25
        + state.source_trust * 0.15
        + state.recency_score * 0.10
        + state.salience * 0.10
        + state.legal_weight * 0.10
        - state.contradiction_pressure * 0.20
    )
    
    return base_similarity_score + boost


def compute_fluid_boost(
    state: FluidMemoryState,
    custom_weights: Optional[dict] = None,
) -> float:
    """
    Compute just the fluid boost value (without applying to score).
    
    Useful for debugging, logging, or custom scoring integration.
    
    Args:
        state: Fluid memory state
        custom_weights: Optional custom weights dict with keys:
            activation, confidence, source_trust, recency_score,
            salience, legal_weight, contradiction_pressure
            
    Returns:
        Computed boost value (can be positive or negative)
    """
    weights = {
        "activation": 0.25,
        "confidence": 0.25,
        "source_trust": 0.15,
        "recency_score": 0.10,
        "salience": 0.10,
        "legal_weight": 0.10,
        "contradiction_pressure": 0.20,  # Applied as negative
    }
    
    if custom_weights:
        weights.update(custom_weights)
    
    boost = (
        state.activation * weights["activation"]
        + state.confidence * weights["confidence"]
        + state.source_trust * weights["source_trust"]
        + state.recency_score * weights["recency_score"]
        + state.salience * weights["salience"]
        + state.legal_weight * weights["legal_weight"]
        - state.contradiction_pressure * weights["contradiction_pressure"]
    )
    
    return boost


def should_boost_retrieval(
    state: FluidMemoryState,
    min_activation: float = 0.1,
    min_confidence: float = 0.3,
) -> bool:
    """
    Determine if a node should get retrieval boost.
    
    Quick check to filter low-quality nodes from boosting.
    
    Args:
        state: Fluid memory state
        min_activation: Minimum activation to qualify
        min_confidence: Minimum confidence to qualify
        
    Returns:
        True if node should be boosted
    """
    # Don't boost nodes with high contradiction pressure
    if state.contradiction_pressure > 0.5:
        return False
    
    # Require some minimum activation or confidence
    if state.activation < min_activation and state.confidence < min_confidence:
        return False
    
    return True
