"""
Judge Memory Models

Data models for the Judge memory subsystem.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any


@dataclass
class EvidenceRecord:
    """
    Immutable evidence record.

    Raw source snapshots are never overwritten.
    Duplicate content (by hash) returns existing record.
    """
    evidence_id: str
    source_type: str
    source_url: Optional[str] = None
    source_title: Optional[str] = None
    retrieved_at: datetime = field(default_factory=datetime.utcnow)
    published_at: Optional[datetime] = None
    jurisdiction: Optional[str] = None
    raw_text: str = ""
    content_hash: str = ""
    storage_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClaimRecord:
    """
    Claim extracted from evidence.

    Every claim must link to valid evidence.
    Claim status derives from source profile unless explicitly set.
    """
    claim_id: str
    evidence_id: str
    claim_text: str
    claim_type: str = "fact"
    subject: Optional[str] = None
    claim_status: str = "needs_verification"
    confidence: float = 0.5
    jurisdiction: Optional[str] = None
    case_id: Optional[str] = None
    judge_id: Optional[str] = None
    person_id: Optional[str] = None
    event_date: Optional[datetime] = None
    source_span: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JudgeMemorySearchResult:
    """
    Search result from Judge memory.

    Includes fluid scoring and explanation if enabled.
    Conflicted results are marked, never hidden by default.
    """
    result_id: str
    result_type: str  # "evidence" | "claim" | "timeline"
    title: str
    summary: str
    score: float
    semantic_score: Optional[float] = None
    graph_score: Optional[float] = None
    fluid_score: Optional[float] = None
    source_trust: Optional[float] = None
    contradiction_pressure: Optional[float] = None
    claim_status: Optional[str] = None
    evidence_ids: List[str] = field(default_factory=list)
    explanation: Dict[str, Any] = field(default_factory=dict)
    is_conflicted: bool = False


@dataclass
class TimelineEvent:
    """
    Timeline event for case/judge/entity.

    Chronological view of memory events.
    """
    event_id: str
    event_type: str
    event_date: Optional[datetime] = None
    title: str = ""
    summary: str = ""
    evidence_ids: List[str] = field(default_factory=list)
    confidence: float = 0.5
    claim_status: str = "needs_verification"


@dataclass
class SourcePacket:
    """
    Source packet with trust profile.

    Explainable source metadata for legal review.
    """
    evidence_id: str
    source_type: str
    source_url: Optional[str] = None
    source_title: Optional[str] = None
    retrieved_at: datetime = field(default_factory=datetime.utcnow)
    content_hash: str = ""
    raw_text_preview: str = ""
    # Structured trust profile
    authority: float = 0.5
    verifiability: float = 0.5
    originality: float = 0.5
    independence: float = 0.5
    legal_status_label: str = "unverified"
    legal_status_weight: float = 0.5
    default_claim_status: str = "needs_verification"
