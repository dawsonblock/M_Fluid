"""Fluid Memory Models"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class MemoryItem(BaseModel):
    """A memory item with fluid state properties."""

    memory_id: str = Field(description="Unique identifier (UUID4)")
    content: str = Field(description="Text content")
    content_hash: str = Field(description="SHA256 hash for deduplication")

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_accessed_at: datetime = Field(default_factory=datetime.utcnow)

    # Access tracking
    access_count: int = Field(default=0, ge=0)

    # Fluid state
    salience: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    volatility: float = Field(default=0.3, ge=0.0, le=1.0)
    stability: float = Field(default=0.5, ge=0.0, le=1.0)
    decay_rate: float = Field(default=0.05, ge=0.0, le=1.0)

    # Decay lanes
    legal_salience: float = Field(default=0.5, ge=0.0, le=1.0)
    trust_salience: float = Field(default=0.5, ge=0.0, le=1.0)
    interest_salience: float = Field(default=0.5, ge=0.0, le=1.0)
    attention_salience: float = Field(default=0.5, ge=0.0, le=1.0)

    # Event tracking
    reinforcement_count: int = Field(default=0, ge=0)
    contradiction_count: int = Field(default=0, ge=0)

    # References
    source_refs: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    links: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def touch(self) -> None:
        """Update access tracking."""
        self.last_accessed_at = datetime.utcnow()
        self.access_count += 1
        self.updated_at = datetime.utcnow()


class RetrievalResult(BaseModel):
    """Result of a memory retrieval operation."""

    memory: MemoryItem
    score: float = Field(ge=0.0, le=1.0)
    match_type: str = Field(default="keyword")  # keyword, semantic, association


class DecayEvent(BaseModel):
    """Record of a decay operation."""

    event_id: str
    memory_id: str
    lane: str  # time, legal, trust, interest, attention
    before_value: float
    after_value: float
    decay_amount: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
