"""
Test Fluid Memory event tracking.
"""

import tempfile
import pytest
from pathlib import Path

from m_flow.memory.fluid.engine import FluidMemoryEngine
from m_flow.memory.fluid.config import FluidMemoryConfig
from m_flow.memory.fluid.events import EventType


@pytest.fixture
def temp_engine():
    """Create a temporary engine for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = FluidMemoryConfig(data_dir=tmpdir)
        engine = FluidMemoryEngine(config)
        yield engine
        engine.close()


def test_add_memory_creates_event(temp_engine):
    """Test that adding memory creates a 'created' event."""
    memory = temp_engine.add_memory("Test content")
    
    events = temp_engine.get_events(memory.memory_id)
    
    assert len(events) >= 1
    assert any(e.event_type == EventType.CREATED for e in events)


def test_get_memory_creates_access_event(temp_engine):
    """Test that retrieving memory creates an 'accessed' event."""
    memory = temp_engine.add_memory("Test content")
    
    # Clear events to isolate access event
    initial_events = temp_engine.get_events(memory.memory_id)
    
    # Get memory (creates access event)
    temp_engine.get_memory(memory.memory_id)
    
    events = temp_engine.get_events(memory.memory_id)
    access_events = [e for e in events if e.event_type == EventType.ACCESSED]
    
    assert len(access_events) >= 1


def test_reinforce_creates_event(temp_engine):
    """Test that reinforcement creates a 'reinforced' event."""
    memory = temp_engine.add_memory("Test content")
    
    temp_engine.reinforce(memory.memory_id, amount=0.2)
    
    events = temp_engine.get_events(memory.memory_id)
    reinforce_events = [e for e in events if e.event_type == EventType.REINFORCED]
    
    assert len(reinforce_events) >= 1


def test_contradict_creates_event(temp_engine):
    """Test that contradiction creates a 'contradicted' event."""
    memory = temp_engine.add_memory("Test content")
    
    temp_engine.contradict(memory.memory_id, amount=0.2)
    
    events = temp_engine.get_events(memory.memory_id)
    contradict_events = [e for e in events if e.event_type == EventType.CONTRADICTED]
    
    assert len(contradict_events) >= 1


def test_mutate_creates_event(temp_engine):
    """Test that mutation creates a 'mutated' event."""
    memory = temp_engine.add_memory("Test content")
    
    temp_engine.mutate(
        memory.memory_id,
        new_content="Updated content",
        reason="Test update",
    )
    
    events = temp_engine.get_events(memory.memory_id)
    mutate_events = [e for e in events if e.event_type == EventType.MUTATED]
    
    assert len(mutate_events) >= 1
    
    # Check event has content change info
    event = mutate_events[0]
    assert "content" in event.delta_json or "state" in event.delta_json


def test_link_creates_event(temp_engine):
    """Test that linking creates a 'linked' event."""
    m1 = temp_engine.add_memory("Source memory")
    m2 = temp_engine.add_memory("Target memory")
    
    temp_engine.link_memories(m1.memory_id, m2.memory_id, link_type="supports")
    
    events = temp_engine.get_events(m1.memory_id)
    link_events = [e for e in events if e.event_type == EventType.LINKED]
    
    assert len(link_events) >= 1
    assert link_events[0].delta_json.get("target_id") == m2.memory_id


def test_decay_creates_events(temp_engine):
    """Test that decay creates 'decayed' events."""
    from time import time
    
    memory = temp_engine.add_memory("Old memory")
    # Manually set updated_at to past
    memory.updated_at = time() - 86400 * 30  # 30 days ago
    temp_engine.storage.update_memory(memory)
    
    # Apply decay
    decayed_count = temp_engine.apply_decay()
    
    events = temp_engine.get_events(memory.memory_id)
    decay_events = [e for e in events if e.event_type == EventType.DECAYED]
    
    assert len(decay_events) >= 1
    assert "old_salience" in decay_events[0].delta_json
    assert "new_salience" in decay_events[0].delta_json


def test_reinforce_event_increases_salience(temp_engine):
    """Test that reinforcement event records salience increase."""
    memory = temp_engine.add_memory("Test content")
    
    temp_engine.reinforce(memory.memory_id, amount=0.2)
    
    events = temp_engine.get_events(memory.memory_id)
    reinforce_event = next(e for e in events if e.event_type == EventType.REINFORCED)
    
    delta = reinforce_event.delta_json
    assert delta["new"]["salience"] > delta["old"]["salience"]


def test_contradict_event_decreases_confidence(temp_engine):
    """Test that contradiction event records confidence decrease."""
    memory = temp_engine.add_memory("Test content")
    
    temp_engine.contradict(memory.memory_id, amount=0.2)
    
    events = temp_engine.get_events(memory.memory_id)
    contradict_event = next(e for e in events if e.event_type == EventType.CONTRADICTED)
    
    delta = contradict_event.delta_json
    assert delta["new"]["confidence"] < delta["old"]["confidence"]


def test_event_timestamps(temp_engine):
    """Test that events have timestamps."""
    memory = temp_engine.add_memory("Test content")
    
    events = temp_engine.get_events(memory.memory_id)
    
    for event in events:
        assert event.timestamp is not None
        assert event.timestamp > 0


def test_event_metadata(temp_engine):
    """Test that events can include metadata."""
    memory = temp_engine.add_memory("Test content")
    
    temp_engine.reinforce(
        memory.memory_id,
        amount=0.1,
        metadata={"source": "user_action", "reason": "important"},
    )
    
    events = temp_engine.get_events(memory.memory_id)
    reinforce_event = next(e for e in events if e.event_type == EventType.REINFORCED)
    
    assert reinforce_event.metadata_json.get("source") == "user_action"
