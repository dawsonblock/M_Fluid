"""
Fluid Memory Models

Core data models for the fluid memory system.
Fluid state is mutable operational state that changes constantly,
in contrast to raw evidence which never changes.
"""

from typing import Optional, List
from time import time
from pydantic import BaseModel, Field


class FluidMemoryState(BaseModel):
    """
    Mutable operational state for a memory node.

    Raw evidence never changes. Graph links change carefully.
    Fluid state changes constantly. Summaries can be regenerated
    from evidence + current state.

    decay_rate is expressed per-day (not per-second).
    Decay lanes:
        short_term = 0.25 / day  (temporary attention)
        normal     = 0.02 / day  (default)
        legal      = 0.002 / day (court/legal evidence)
    """
    node_id: str
    activation: float = 0.0
    confidence: float = 0.5
    source_trust: float = 0.5
    recency_score: float = 1.0
    decay_rate: float = 0.02          # per-day; default = NORMAL_DECAY
    decay_lane: str = "normal"        # "short_term" | "normal" | "legal"
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
    touched_node_ids: List[str]
    source_id: Optional[str] = None
    source_type: Optional[str] = None
    source_trust: float = 0.5
    salience: float = 0.5
    legal_weight: float = 0.0
    decay_lane: str = "normal"
    supports: List[str] = Field(default_factory=list)
    contradicts: List[str] = Field(default_factory=list)


class ClaimConflict(BaseModel):
    """
    Records a detected conflict between two memory nodes.

    Stored in the fluid_claim_conflicts table.
    Created by ContradictionDetector when an LLM detects
    that two nodes assert incompatible facts.
    """
    node_id_a: str
    node_id_b: str
    source_id_a: Optional[str] = None
    source_id_b: Optional[str] = None
    conflict_reason: str = ""
    confidence: float = 0.0           # LLM confidence in conflict detection
    detected_at: float = Field(default_factory=time)


# ---------------------------------------------------------------------------
# Backward-compat shim — use SourceRegistry for new code
# ---------------------------------------------------------------------------

_TRUST_FALLBACK = {
    "court_record": (0.95, 1.00),
    "government_data": (0.85, 0.80),
    "police_release": (0.80, 0.70),
    "academic_paper": (0.75, 0.40),
    "expert_report": (0.70, 0.50),
    "witness_statement": (0.50, 0.40),
    "mainstream_news": (0.60, 0.30),
    "blog_social": (0.25, 0.05),
    "unknown": (0.10, 0.00),
}


def get_source_weights(source_type: Optional[str]) -> tuple[float, float]:
    """
    Synchronous fallback for trust/legal weights.

    Prefer SourceRegistry.get_weights() for new async code.
    This shim reads from the hardcoded defaults and is used only
    in synchronous contexts (e.g., episodic write helper).
    """
    key = (source_type or "unknown").lower().replace(" ", "_")
    return _TRUST_FALLBACK.get(key, (0.10, 0.00))
