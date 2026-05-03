"""
Fluid Memory Models

Core data models for the fluid memory system.
Fluid state is mutable operational state that changes constantly,
in contrast to raw evidence which never changes.

Architecture principle — Activation ≠ Truth:
    Activation reflects recency of attention (social amplification, user access).
    Trust derives from provenance, cross-confirmation, and legal authority.
    These are separate fields with separate decay curves.

Layer order:
    RAW SOURCE → EXTRACTED CLAIMS → EVIDENCE LINKS → ENTITY GRAPH
    → FLUID STATE ENGINE → RETRIEVAL + SCORING
"""

from typing import Optional, List
from time import time
from pydantic import BaseModel, Field


class FluidMemoryState(BaseModel):
    """
    Mutable operational state for a memory node.

    Raw evidence never changes. Fluid state changes constantly.

    Decay lanes (5-lane model):
        attention    = 0.20 / day  (short-term visibility, breaking news)
        interest     = 0.05 / day  (user/session engagement)
        trust        = 0.000 / day (provenance trust — IMMUTABLE)
        legal        = 0.000 / day (court/gov evidence — IMMUTABLE)
        contradiction = 0.01 / day (conflict pressure eases toward 0)

    JudgeTracker / crime-mapping fields:
        source_lineage          chain of contributing source IDs
        jurisdiction            ISO 3166-2 or custom jurisdiction code
        judge_id                linked judge node_id
        event_confidence        cross-confirmation confidence [0-1]
        geographic_scope        "local" | "regional" | "national"
        contradiction_cluster_id UUID grouping related conflicts
        media_amplification     0-1, echo/duplicate detection score
    """
    node_id: str
    activation: float = 0.0
    confidence: float = 0.5
    source_trust: float = 0.5
    recency_score: float = 1.0
    decay_rate: float = 0.05          # per-day; default = INTEREST_DECAY
    decay_lane: str = "interest"      # "attention"|"interest"|"trust"|"legal"|"contradiction"
    reinforcement_count: int = 0
    contradiction_pressure: float = 0.0
    salience: float = 0.5
    user_relevance: float = 0.5
    legal_weight: float = 0.0
    last_touched_at: float = Field(default_factory=time)

    # ---------------------------------------------------------------------------
    # Provenance / lineage
    # ---------------------------------------------------------------------------
    source_lineage: List[str] = Field(default_factory=list)
    media_amplification: float = 0.0   # 0-1; how much media echo detected

    # ---------------------------------------------------------------------------
    # Legal / jurisdiction (JudgeTracker)
    # ---------------------------------------------------------------------------
    jurisdiction: Optional[str] = None          # e.g. "US-TX", "federal", "unknown"
    judge_id: Optional[str] = None              # linked judge node_id
    event_confidence: float = 0.5              # cross-confirmation confidence [0-1]

    # ---------------------------------------------------------------------------
    # Geographic
    # ---------------------------------------------------------------------------
    geographic_scope: Optional[str] = None     # "local" | "regional" | "national"

    # ---------------------------------------------------------------------------
    # Contradiction clusters
    # ---------------------------------------------------------------------------
    contradiction_cluster_id: Optional[str] = None  # UUID shared across conflicting pair


class FluidUpdateEvent(BaseModel):
    """
    Event that triggers a fluid memory update.

    Created when episodic memory writes occur; carries source metadata
    for trust/salience weighting and JudgeTracker enrichment.
    """
    touched_node_ids: List[str]
    source_id: Optional[str] = None
    source_type: Optional[str] = None
    source_trust: float = 0.5
    salience: float = 0.5
    legal_weight: float = 0.0
    decay_lane: str = "interest"
    supports: List[str] = Field(default_factory=list)
    contradicts: List[str] = Field(default_factory=list)

    # JudgeTracker enrichment (optional — populated by write_episodic_memories)
    jurisdiction: Optional[str] = None
    judge_id: Optional[str] = None
    geographic_scope: Optional[str] = None
    event_confidence: float = 0.5
    parent_source_ids: List[str] = Field(default_factory=list)  # for lineage chain


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
    contradiction_cluster_id: Optional[str] = None  # shared UUID across conflict group
    conflict_status: str = "needs_review"  # confirmed_conflict, possible_conflict, needs_review, resolved, rejected


class SourceLineageRecord(BaseModel):
    """
    Tracks the citation / confirmation / contradiction relationship
    between two source documents contributing to a node.

    Stored in the fluid_source_lineage table.
    Used by CitationGraph to build the evidence chain.
    """
    node_id: str
    parent_source_id: str
    child_source_id: str
    relationship: str = "cites"       # "cites" | "confirms" | "contradicts" | "amplifies"
    recorded_at: float = Field(default_factory=time)


class MediaAmplificationEvent(BaseModel):
    """
    Records detected duplicate articles collapsing into a single node.

    When multiple articles say the same thing, media_amplification rises.
    High amplification means a node is socially amplified, not necessarily true.
    Stored in the fluid_media_amplification table.
    """
    node_id: str
    canonical_source_id: str
    duplicate_source_ids: List[str] = Field(default_factory=list)
    amplification_factor: float = 0.0   # normalized count of duplicates [0-1]
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
