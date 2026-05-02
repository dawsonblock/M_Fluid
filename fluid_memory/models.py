"""Fluid Memory Models"""

import hashlib
import uuid
from time import time
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic import ConfigDict


def _new_memory_id() -> str:
    return f"mem_{uuid.uuid4().hex[:16]}"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_VALID_LINK_TYPES = {"supports", "contradicts", "related", "causes", "depends_on"}


class MemoryItem(BaseModel):
    """A memory item with fluid state properties."""

    model_config = ConfigDict(validate_assignment=True)

    memory_id: str = Field(default_factory=_new_memory_id, description="Unique identifier")
    content: str = Field(description="Text content")
    content_hash: str = Field(default="", description="SHA256 hash for deduplication")

    # Timestamps (unix float)
    created_at: float = Field(default_factory=time)
    updated_at: float = Field(default_factory=time)
    last_accessed_at: Optional[float] = Field(default=None)

    # Access tracking
    access_count: int = Field(default=0)

    # Fluid state — stored as plain floats; validators clamp on construction/assignment
    salience: float = Field(default=0.5)
    confidence: float = Field(default=0.5)
    volatility: float = Field(default=0.3)
    stability: float = Field(default=0.5)
    decay_rate: float = Field(default=0.05)

    # Decay lanes
    legal_salience: float = Field(default=0.5)
    trust_salience: float = Field(default=0.5)
    interest_salience: float = Field(default=0.5)
    attention_salience: float = Field(default=0.5)

    # Event tracking
    reinforcement_count: int = Field(default=0)
    contradiction_count: int = Field(default=0)

    # References
    source_refs: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    links: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _set_content_hash(self) -> "MemoryItem":
        if not self.content_hash:
            object.__setattr__(self, "content_hash", _sha256(self.content))
        return self

    @field_validator("salience", "confidence", "volatility", "stability", "decay_rate",
                     "legal_salience", "trust_salience", "interest_salience",
                     "attention_salience", mode="before")
    @classmethod
    def _clamp_float(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))

    def touch(self) -> None:
        """Update access tracking."""
        now = time()
        self.last_accessed_at = now
        self.access_count += 1
        self.updated_at = now


class MemoryLink(BaseModel):
    """A directed link between two memory items."""

    model_config = ConfigDict(validate_assignment=True)

    link_id: str = Field(default_factory=lambda: f"lnk_{uuid.uuid4().hex[:12]}")
    source_memory_id: str
    target_memory_id: str
    link_type: str = Field(default="related")
    strength: float = Field(default=0.5)

    @field_validator("strength", mode="before")
    @classmethod
    def _clamp_strength(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))

    @field_validator("link_type", mode="before")
    @classmethod
    def _validate_link_type(cls, v: str) -> str:
        if v not in _VALID_LINK_TYPES:
            raise ValueError(
                f"link_type must be one of {sorted(_VALID_LINK_TYPES)}, got {v!r}"
            )
        return v


class RetrievalResult(BaseModel):
    """Result of a memory retrieval operation."""

    model_config = ConfigDict(validate_assignment=True)

    memory: MemoryItem
    score: float = Field(default=0.0)
    match_type: str = Field(default="keyword")  # keyword, semantic, association

    @field_validator("score", mode="before")
    @classmethod
    def _clamp_score(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))


class DecayEvent(BaseModel):
    """Record of a decay operation."""

    event_id: str
    memory_id: str
    lane: str  # time, legal, trust, interest, attention
    before_value: float
    after_value: float
    decay_amount: float
    timestamp: float = Field(default_factory=time)
