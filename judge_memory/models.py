"""Judge Memory Models"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class EvidenceRecord(BaseModel):
    """Immutable evidence record with source metadata.
    
    Evidence is stored immutably - never modified after ingestion.
    Duplicates are detected by content hash and return existing record.
    """
    
    evidence_id: str = Field(description="Unique identifier (UUID)")
    content_hash: str = Field(description="SHA256 hash of content")
    source_type: str = Field(description="Type: court_record, government_data, etc.")
    source_url: Optional[str] = Field(default=None, description="Source URL if available")
    source_title: Optional[str] = Field(default=None, description="Human-readable title")
    content_preview: Optional[str] = Field(default=None, description="First 1000 chars of content for search")
    jurisdiction: Optional[str] = Field(default=None, description="Jurisdiction code")
    published_at: Optional[datetime] = Field(default=None, description="Publication date")
    file_path: Optional[str] = Field(default=None, description="Path to stored file")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    ingested_at: datetime = Field(default_factory=datetime.utcnow, description="Ingestion timestamp")


class ClaimRecord(BaseModel):
    """Mutable claim linked to evidence.
    
    Claims are interpretations or assertions based on evidence.
    Unlike evidence, claims can be updated, reviewed, and contradicted.
    """
    
    claim_id: str = Field(description="Unique identifier (UUID)")
    evidence_id: str = Field(description="Parent evidence record ID")
    claim_text: str = Field(description="The claim content")
    claim_type: str = Field(default="fact", description="Type: fact, ruling, opinion, etc.")
    case_id: Optional[str] = Field(default=None, description="Associated case ID")
    judge_id: Optional[str] = Field(default=None, description="Associated judge ID")
    person_id: Optional[str] = Field(default=None, description="Associated person/entity ID")
    entity_id: Optional[str] = Field(default=None, description="Associated entity ID")
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence score 0.0-1.0"
    )
    status: str = Field(
        default="active",
        description="Status: active, under_review, retracted, confirmed"
    )
    tags: List[str] = Field(default_factory=list, description="Categorization tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")


class TimelineEvent(BaseModel):
    """Chronological event for case/person/judge timeline."""
    
    event_id: str = Field(description="Unique identifier")
    event_type: str = Field(description="Type: ruling, hearing, filing, etc.")
    event_date: datetime = Field(description="When the event occurred")
    description: str = Field(description="Event description")
    evidence_id: Optional[str] = Field(default=None, description="Associated evidence")
    claim_id: Optional[str] = Field(default=None, description="Associated claim")
    case_id: Optional[str] = Field(default=None, description="Associated case")
    judge_id: Optional[str] = Field(default=None, description="Associated judge")
    person_id: Optional[str] = Field(default=None, description="Associated person")
    entity_id: Optional[str] = Field(default=None, description="Associated entity")
    jurisdiction: Optional[str] = Field(default=None, description="Jurisdiction")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SourcePacket(BaseModel):
    """Explainable source trust profile.
    
    Provides transparency for legal review by exposing
    how source authority and trust are calculated.
    """
    
    evidence_id: str = Field(description="Source evidence ID")
    authority: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Source authority score"
    )
    verifiability: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="How verifiable the source is"
    )
    originality: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Originality vs derivative"
    )
    independence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Independence from bias"
    )
    legal_status_label: str = Field(
        default="unverified",
        description="Legal classification"
    )
    legal_status_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Legal status numerical weight"
    )
    default_claim_status: str = Field(
        default="needs_verification",
        description="Default status for claims from this source"
    )
    source_type: Optional[str] = Field(default=None, description="Source type")
    source_url: Optional[str] = Field(default=None, description="Source URL")


class JudgeMemorySearchResult(BaseModel):
    """Search result combining evidence and claims."""
    
    result_type: str = Field(
        description="Type: evidence or claim"
    )
    record_id: str = Field(description="Evidence or claim ID")
    title: Optional[str] = Field(default=None, description="Display title")
    content_preview: str = Field(description="Content snippet")
    source_type: Optional[str] = Field(default=None, description="Source type")
    jurisdiction: Optional[str] = Field(default=None, description="Jurisdiction")
    confidence: Optional[float] = Field(default=None, description="Confidence score")
    status: Optional[str] = Field(default=None, description="Claim status if applicable")
    score: float = Field(default=0.0, description="Search relevance score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional data")
