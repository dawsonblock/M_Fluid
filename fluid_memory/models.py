"""
Fluid Memory Models

Core data models for memory items and retrieval results.
"""

import hashlib
import uuid
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator
from time import time


class MemoryItem(BaseModel):
    """
    A memory item with fluid state.
    
    Memories have state that shifts over time when touched by new input,
    reuse, contradictions, confirmations, age, and importance.
    
    Fields:
        memory_id: Unique identifier
        content: Text content of the memory
        content_hash: MD5 hash of content for duplicate detection
        created_at: Unix timestamp when memory was created
        updated_at: Unix timestamp of last modification
        last_accessed_at: Unix timestamp of last access (None if never accessed)
        access_count: Number of times memory was retrieved
        salience: How important the memory currently is [0.0, 1.0]
        confidence: How reliable the memory appears [0.0, 1.0]
        volatility: How likely memory is to change [0.0, 1.0]
        stability: How resistant to decay/mutation [0.0, 1.0]
        decay_rate: How quickly salience decreases [0.0, 1.0]
        reinforcement_count: Number of confirming events
        contradiction_count: Number of conflicting events
        source_refs: List of source references
        tags: List of tags for categorization
        links: List of linked memory IDs
        metadata: Additional arbitrary metadata
    """
    
    memory_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    content_hash: str = ""
    created_at: float = Field(default_factory=time)
    updated_at: float = Field(default_factory=time)
    last_accessed_at: Optional[float] = None
    access_count: int = 0
    salience: float = 0.5
    confidence: float = 0.5
    volatility: float = 0.3
    stability: float = 0.5
    decay_rate: float = 0.05
    reinforcement_count: int = 0
    contradiction_count: int = 0
    source_refs: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    links: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator("salience", "confidence", "volatility", "stability", "decay_rate")
    @classmethod
    def clamp_0_1(cls, v: float) -> float:
        """Clamp float fields to [0.0, 1.0]."""
        return max(0.0, min(1.0, v))
    
    @model_validator(mode="after")
    def compute_content_hash(self):
        """Compute MD5 hash of content if not set."""
        if not self.content_hash and self.content:
            self.content_hash = hashlib.md5(self.content.encode("utf-8")).hexdigest()
        return self
    
    def touch(self) -> None:
        """Update access-related fields."""
        self.last_accessed_at = time()
        self.access_count += 1
    
    def update_timestamp(self) -> None:
        """Update the modification timestamp."""
        self.updated_at = time()


class RetrievalResult(BaseModel):
    """
    Result of a memory retrieval operation.
    
    Contains the memory and computed retrieval score.
    """
    
    memory: MemoryItem
    score: float
    match_type: str = "text"  # text, tag, or combined
    
    @field_validator("score")
    @classmethod
    def clamp_score(cls, v: float) -> float:
        """Clamp score to [0.0, 1.0]."""
        return max(0.0, min(1.0, v))


class MemoryLink(BaseModel):
    """
    A link between two memories.
    """
    
    link_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_memory_id: str
    target_memory_id: str
    link_type: str = "related"  # related, supports, contradicts, parent, child, sequence
    strength: float = 0.5
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time)
    
    @field_validator("strength")
    @classmethod
    def clamp_strength(cls, v: float) -> float:
        """Clamp strength to [0.0, 1.0]."""
        return max(0.0, min(1.0, v))
    
    @field_validator("link_type")
    @classmethod
    def validate_link_type(cls, v: str) -> str:
        """Validate link type is one of allowed values."""
        allowed = {"related", "supports", "contradicts", "parent", "child", "sequence"}
        if v not in allowed:
            raise ValueError(f"link_type must be one of {allowed}")
        return v
