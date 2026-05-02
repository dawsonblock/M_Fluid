"""
Test Fluid Memory storage operations.
"""

import tempfile
import pytest
from pathlib import Path

from m_flow.memory.fluid.storage import MemoryStorage
from m_flow.memory.fluid.models import MemoryItem, MemoryLink
from m_flow.memory.fluid.events import MemoryEvent, EventType


@pytest.fixture
def temp_storage():
    """Create a temporary storage for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        storage = MemoryStorage(str(db_path))
        yield storage


def test_save_and_get_memory(temp_storage):
    """Test saving and retrieving a memory."""
    memory = MemoryItem(content="Test content", tags=["tag1"])
    temp_storage.save_memory(memory)
    
    retrieved = temp_storage.get_memory(memory.memory_id)
    assert retrieved is not None
    assert retrieved.content == "Test content"
    assert retrieved.tags == ["tag1"]


def test_get_memory_by_hash(temp_storage):
    """Test retrieving memory by content hash."""
    memory = MemoryItem(content="Unique content")
    temp_storage.save_memory(memory)
    
    retrieved = temp_storage.get_memory_by_hash(memory.content_hash)
    assert retrieved is not None
    assert retrieved.memory_id == memory.memory_id


def test_update_memory(temp_storage):
    """Test updating a memory."""
    memory = MemoryItem(content="Original")
    temp_storage.save_memory(memory)
    
    memory.content = "Updated"
    memory.salience = 0.9
    temp_storage.update_memory(memory)
    
    retrieved = temp_storage.get_memory(memory.memory_id)
    assert retrieved.content == "Updated"
    assert retrieved.salience == 0.9


def test_delete_memory(temp_storage):
    """Test deleting a memory."""
    memory = MemoryItem(content="To delete")
    temp_storage.save_memory(memory)
    
    # Verify it exists
    assert temp_storage.get_memory(memory.memory_id) is not None
    
    # Delete it
    temp_storage.delete_memory(memory.memory_id)
    
    # Verify it's gone
    assert temp_storage.get_memory(memory.memory_id) is None


def test_search_memories_text(temp_storage):
    """Test text-based search."""
    m1 = MemoryItem(content="Python programming")
    m2 = MemoryItem(content="JavaScript coding")
    m3 = MemoryItem(content="Data science with Python")
    
    temp_storage.save_memory(m1)
    temp_storage.save_memory(m2)
    temp_storage.save_memory(m3)
    
    results = temp_storage.search_memories(query="Python", limit=10)
    assert len(results) == 2
    contents = [r.content for r in results]
    assert "Python programming" in contents
    assert "Data science with Python" in contents


def test_search_memories_tags(temp_storage):
    """Test tag-based search."""
    m1 = MemoryItem(content="Content 1", tags=["python", "coding"])
    m2 = MemoryItem(content="Content 2", tags=["javascript"])
    m3 = MemoryItem(content="Content 3", tags=["python"])
    
    temp_storage.save_memory(m1)
    temp_storage.save_memory(m2)
    temp_storage.save_memory(m3)
    
    results = temp_storage.search_memories(tags=["python"], limit=10)
    assert len(results) == 2


def test_save_and_get_event(temp_storage):
    """Test saving and retrieving events."""
    memory = MemoryItem(content="Test")
    temp_storage.save_memory(memory)
    
    event = MemoryEvent(
        event_id="evt-1",
        memory_id=memory.memory_id,
        event_type=EventType.CREATED,
        delta_json={"key": "value"},
    )
    temp_storage.save_event(event)
    
    events = temp_storage.get_events(memory.memory_id)
    assert len(events) == 1
    assert events[0].event_type == EventType.CREATED


def test_save_and_get_link(temp_storage):
    """Test saving and retrieving links."""
    m1 = MemoryItem(content="Source")
    m2 = MemoryItem(content="Target")
    temp_storage.save_memory(m1)
    temp_storage.save_memory(m2)
    
    link = MemoryLink(
        source_memory_id=m1.memory_id,
        target_memory_id=m2.memory_id,
        link_type="supports",
        strength=0.8,
    )
    temp_storage.save_link(link)
    
    links = temp_storage.get_links(m1.memory_id)
    assert len(links) == 1
    assert links[0].link_type == "supports"


def test_persistence(temp_storage):
    """Test that data persists across storage instances."""
    memory = MemoryItem(content="Persistent content")
    temp_storage.save_memory(memory)
    
    # Create new storage pointing to same DB
    new_storage = MemoryStorage(temp_storage.db_path)
    retrieved = new_storage.get_memory(memory.memory_id)
    
    assert retrieved is not None
    assert retrieved.content == "Persistent content"
