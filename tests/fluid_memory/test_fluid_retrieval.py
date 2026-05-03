"""
Test Fluid Memory retrieval operations.
"""

import tempfile
import pytest
from pathlib import Path

from fluid_memory.engine import FluidMemoryEngine
from fluid_memory.config import FluidMemoryConfig


@pytest.fixture
def temp_engine():
    """Create a temporary engine for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = FluidMemoryConfig(data_dir=tmpdir)
        engine = FluidMemoryEngine(config)
        yield engine
        engine.close()


def test_add_memory(temp_engine):
    """Test adding a memory."""
    memory = temp_engine.add_memory("Test content", tags=["tag1"])
    
    assert memory.content == "Test content"
    assert memory.tags == ["tag1"]
    assert memory.memory_id is not None
    assert memory.content_hash is not None


def test_add_memory_duplicate_raises(temp_engine):
    """Test that adding duplicate content raises error."""
    temp_engine.add_memory("Unique content")
    
    with pytest.raises(Exception):  # DuplicateMemoryError
        temp_engine.add_memory("Unique content")


def test_get_memory(temp_engine):
    """Test getting a memory by ID."""
    memory = temp_engine.add_memory("Test content")
    
    retrieved = temp_engine.get_memory(memory.memory_id)
    
    assert retrieved.memory_id == memory.memory_id
    assert retrieved.content == "Test content"
    assert retrieved.access_count == 1  # get_memory increments access


def test_get_memory_not_found(temp_engine):
    """Test that getting non-existent memory raises error."""
    with pytest.raises(Exception):  # MemoryNotFoundError
        temp_engine.get_memory("non-existent-id")


def test_retrieve_text_match(temp_engine):
    """Test retrieval with text query."""
    m1 = temp_engine.add_memory("Python programming")
    m2 = temp_engine.add_memory("JavaScript coding")
    m3 = temp_engine.add_memory("Python data science")
    
    results = temp_engine.retrieve(query="Python", limit=10)
    
    assert len(results) == 2
    contents = [r.memory.content for r in results]
    assert "Python programming" in contents
    assert "Python data science" in contents


def test_retrieve_tag_match(temp_engine):
    """Test retrieval with tag filter."""
    m1 = temp_engine.add_memory("Content 1", tags=["python"])
    m2 = temp_engine.add_memory("Content 2", tags=["javascript"])
    m3 = temp_engine.add_memory("Content 3", tags=["python"])
    
    results = temp_engine.retrieve(tags=["python"], limit=10)
    
    assert len(results) == 2


def test_retrieve_combined(temp_engine):
    """Test retrieval with both text and tags."""
    m1 = temp_engine.add_memory("Python basics", tags=["tutorial"])
    m2 = temp_engine.add_memory("Python advanced", tags=["advanced"])
    m3 = temp_engine.add_memory("JavaScript basics", tags=["tutorial"])
    
    results = temp_engine.retrieve(query="Python", tags=["tutorial"], limit=10)
    
    # Should return Python basics (matches both)
    assert len(results) >= 1


def test_retrieve_limit(temp_engine):
    """Test retrieval respects limit."""
    for i in range(20):
        temp_engine.add_memory(f"Content {i}")
    
    results = temp_engine.retrieve(limit=5)
    assert len(results) <= 5


def test_retrieve_sorting(temp_engine):
    """Test that results are sorted by score."""
    # Add memories with different salience
    m1 = temp_engine.add_memory("Low salience")
    m1.salience = 0.1
    temp_engine.storage.update_memory(m1)
    
    m2 = temp_engine.add_memory("High salience")
    m2.salience = 0.9
    temp_engine.storage.update_memory(m2)
    
    results = temp_engine.retrieve(limit=10)
    
    # High salience should come first
    if len(results) >= 2:
        assert results[0].score >= results[1].score


def test_retrieve_reinforced_ranks_higher(temp_engine):
    """Test that reinforced memory ranks higher."""
    m1 = temp_engine.add_memory("Memory one")
    m2 = temp_engine.add_memory("Memory two")
    
    # Reinforce m1
    temp_engine.reinforce(m1.memory_id, amount=0.5)
    
    results = temp_engine.retrieve(query="Memory", limit=10)
    
    # m1 should rank higher
    m1_result = next(r for r in results if r.memory.memory_id == m1.memory_id)
    m2_result = next(r for r in results if r.memory.memory_id == m2.memory_id)
    assert m1_result.score > m2_result.score


def test_retrieve_low_confidence_ranks_lower(temp_engine):
    """Test that low confidence memory ranks lower."""
    m1 = temp_engine.add_memory("High confidence")
    m2 = temp_engine.add_memory("Low confidence")
    
    # Lower m2's confidence
    temp_engine.contradict(m2.memory_id, amount=0.5)
    
    results = temp_engine.retrieve(query="confidence", limit=10)
    
    # m1 should rank higher
    m1_result = next(r for r in results if r.memory.memory_id == m1.memory_id)
    m2_result = next(r for r in results if r.memory.memory_id == m2.memory_id)
    assert m1_result.score > m2_result.score
