"""
Test Fluid Memory state models.
"""

import pytest
from fluid_memory.models import MemoryItem, RetrievalResult, MemoryLink
from fluid_memory.events import EventType, MemoryEvent
from fluid_memory.state import clamp01


def test_clamp01():
    """Test clamping function."""
    assert clamp01(0.5) == 0.5
    assert clamp01(-0.1) == 0.0
    assert clamp01(1.5) == 1.0
    assert clamp01(0.0) == 0.0
    assert clamp01(1.0) == 1.0


def test_memory_item_defaults():
    """Test MemoryItem default values."""
    item = MemoryItem(content="Test content")
    
    assert item.content == "Test content"
    assert item.salience == 0.5
    assert item.confidence == 0.5
    assert item.volatility == 0.3
    assert item.stability == 0.5
    assert item.decay_rate == 0.05
    assert item.access_count == 0
    assert item.reinforcement_count == 0
    assert item.contradiction_count == 0
    assert item.tags == []
    assert item.source_refs == []
    assert item.links == []
    assert item.metadata == {}


def test_memory_item_content_hash():
    """Test that content hash is computed automatically."""
    item = MemoryItem(content="Test content")
    assert item.content_hash is not None
    assert len(item.content_hash) == 32  # MD5 is 32 hex chars
    
    # Same content should produce same hash
    item2 = MemoryItem(content="Test content")
    assert item.content_hash == item2.content_hash
    
    # Different content should produce different hash
    item3 = MemoryItem(content="Different content")
    assert item.content_hash != item3.content_hash


def test_memory_item_field_clamping():
    """Test that fields are clamped to valid ranges."""
    item = MemoryItem(
        content="Test",
        salience=1.5,
        confidence=-0.5,
        volatility=2.0,
        stability=-1.0,
        decay_rate=5.0,
    )
    
    assert item.salience == 1.0
    assert item.confidence == 0.0
    assert item.volatility == 1.0
    assert item.stability == 0.0
    assert item.decay_rate == 1.0


def test_memory_item_touch():
    """Test touch method updates access fields."""
    item = MemoryItem(content="Test")
    
    assert item.access_count == 0
    assert item.last_accessed_at is None
    
    item.touch()
    
    assert item.access_count == 1
    assert item.last_accessed_at is not None
    
    item.touch()
    assert item.access_count == 2


def test_retrieval_result():
    """Test RetrievalResult model."""
    memory = MemoryItem(content="Test")
    result = RetrievalResult(memory=memory, score=0.75, match_type="text")
    
    assert result.memory == memory
    assert result.score == 0.75
    assert result.match_type == "text"


def test_retrieval_result_score_clamping():
    """Test that score is clamped to [0.0, 1.0]."""
    memory = MemoryItem(content="Test")
    result = RetrievalResult(memory=memory, score=1.5)
    assert result.score == 1.0
    
    result2 = RetrievalResult(memory=memory, score=-0.5)
    assert result2.score == 0.0


def test_memory_link():
    """Test MemoryLink model."""
    link = MemoryLink(
        source_memory_id="source-1",
        target_memory_id="target-1",
        link_type="supports",
        strength=0.8,
    )
    
    assert link.source_memory_id == "source-1"
    assert link.target_memory_id == "target-1"
    assert link.link_type == "supports"
    assert link.strength == 0.8


def test_memory_link_strength_clamping():
    """Test that link strength is clamped."""
    link = MemoryLink(
        source_memory_id="s1",
        target_memory_id="t1",
        strength=1.5,
    )
    assert link.strength == 1.0
    
    link2 = MemoryLink(
        source_memory_id="s1",
        target_memory_id="t1",
        strength=-0.5,
    )
    assert link2.strength == 0.0


def test_memory_link_type_validation():
    """Test that invalid link types are rejected."""
    with pytest.raises(ValueError):
        MemoryLink(
            source_memory_id="s1",
            target_memory_id="t1",
            link_type="invalid_type",
        )


def test_memory_event():
    """Test MemoryEvent model."""
    event = MemoryEvent(
        event_id="evt-1",
        memory_id="mem-1",
        event_type=EventType.CREATED,
        delta_json={"key": "value"},
        metadata_json={"source": "test"},
    )
    
    assert event.event_id == "evt-1"
    assert event.memory_id == "mem-1"
    assert event.event_type == EventType.CREATED
    assert event.delta_json == {"key": "value"}
    assert event.metadata_json == {"source": "test"}
    assert event.timestamp is not None
