"""
Fluid Memory Models

Core data models for the fluid memory system.
Fluid state is mutable operational state that changes constantly,
in contrast to raw evidence which never changes.
"""

from typing import Optional
from time import time
from pydantic import BaseModel, Field


class FluidMemoryState(BaseModel):
    """
    Mutable operational state for a memory node.
    
    This represents the "fluid" properties of memory that change over time
    based on access patterns, source quality, contradictions, and decay.
    
    Raw evidence never changes. Graph links change carefully.
    Fluid state changes constantly. Summaries can be regenerated
    from evidence + current state.
    """
    node_id: str
    activation: float = 0.0
    confidence: float = 0.5
    source_trust: float = 0.5
    recency_score: float = 1.0
    decay_rate: float = 0.01
    reinforcement_count: int = 0
    contradiction_pressure: float = 0.0
    salience: float = 0.5
    user_relevance: float = 0.5
    legal_weight: float = 0.0
    last_touched_at: float = Field(default_factory=time)


class FluidUpdateEvent(BaseModel):
    """
    Event that triggers a fluid memory update.
    
    Created when episodic memory writes occur, this event
    carries source metadata for trust/salience weighting.
    """
    touched_node_ids: list[str]
    source_id: Optional[str] = None
    source_type: Optional[str] = None
    source_trust: float = 0.5
    salience: float = 0.5
    legal_weight: float = 0.0
    supports: list[str] = []
    contradicts: list[str] = []


# Source trust weights for common source types
# Usage: source_trust = SOURCE_TRUST_WEIGHTS.get(source_type, 0.1)
SOURCE_TRUST_WEIGHTS = {
    "court_record": 0.95,
    "police_release": 0.80,
    "government_data": 0.85,
    "mainstream_news": 0.60,
    "blog_social": 0.25,
    "unknown": 0.10,
}

LEGAL_WEIGHT_WEIGHTS = {
    "court_record": 1.00,
    "police_release": 0.70,
    "government_data": 0.80,
    "mainstream_news": 0.30,
    "blog_social": 0.05,
    "unknown": 0.00,
}


def get_source_weights(source_type: Optional[str]) -> tuple[float, float]:
    """
    Get trust and legal weights for a source type.
    
    Args:
        source_type: The type of source (e.g., "court_record", "mainstream_news")
        
    Returns:
        Tuple of (source_trust, legal_weight)
    """
    normalized = (source_type or "unknown").lower().replace(" ", "_")
    trust = SOURCE_TRUST_WEIGHTS.get(normalized, 0.10)
    legal = LEGAL_WEIGHT_WEIGHTS.get(normalized, 0.00)
    return trust, legal
