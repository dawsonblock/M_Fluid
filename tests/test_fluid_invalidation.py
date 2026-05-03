"""
Test Fluid Memory invalidation at retrieval boundary (Phase 3).

- Invalidated memory does not appear in ordinary retrieval
- Invalidated memory appears in admin retrieval
- Direct ID fetch behavior is explicit
- Invalidation does not delete the original record
"""

import tempfile

import pytest
from fluid_memory import FluidMemoryEngine, FluidMemoryConfig
from fluid_memory.models import MemoryItem
from fluid_memory.storage import MemoryStorage


def test_invalidated_memory_hidden_from_get_memory():
    """Test that get_memory returns None for invalidated memories by default."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = MemoryStorage(f"{tmpdir}/test.db")
        # Create and save a memory
        memory = MemoryItem(content="Test content")
        storage.save_memory(memory)

        # Invalidate the memory
        result = storage.invalidate(memory.memory_id, reason="Test invalidation")
        assert result is True

        # Normal get_memory should return None
        retrieved = storage.get_memory(memory.memory_id)
        assert retrieved is None

        # With include_invalidated=True, should return the memory
        retrieved = storage.get_memory(memory.memory_id, include_invalidated=True)
        assert retrieved is not None
        assert retrieved.memory_id == memory.memory_id
        assert retrieved.content == "Test content"


def test_invalidated_memory_hidden_from_search():
    """Test that search excludes invalidated memories by default."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = MemoryStorage(f"{tmpdir}/test.db")
        # Create memories
        memory1 = MemoryItem(content="Apple pie recipe")
        memory2 = MemoryItem(content="Apple computer manual")
        storage.save_memory(memory1)
        storage.save_memory(memory2)

        # Search should find both
        results = storage.search_memories("Apple", limit=10)
        assert len(results) == 2

        # Invalidate one
        storage.invalidate(memory1.memory_id, reason="Outdated recipe")

        # Search should only find the valid one
        results = storage.search_memories("Apple", limit=10)
        assert len(results) == 1
        assert results[0].memory_id == memory2.memory_id

        # Search with include_invalidated should find both
        results = storage.search_memories("Apple", limit=10, include_invalidated=True)
        assert len(results) == 2


def test_invalidated_memory_hidden_from_get_all():
    """Test that get_all excludes invalidated memories by default."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = MemoryStorage(f"{tmpdir}/test.db")
        # Create memories
        memories = [
            MemoryItem(content=f"Memory {i}")
            for i in range(3)
        ]
        for m in memories:
            storage.save_memory(m)

        # get_all should return all 3
        results = storage.get_all(limit=10)
        assert len(results) == 3

        # Invalidate one
        storage.invalidate(memories[1].memory_id, reason="Test")

        # get_all should only return 2
        results = storage.get_all(limit=10)
        assert len(results) == 2
        returned_ids = {m.memory_id for m in results}
        assert memories[1].memory_id not in returned_ids

        # get_all with include_invalidated should return all 3
        results = storage.get_all(limit=10, include_invalidated=True)
        assert len(results) == 3


def test_get_invalidated_memories_returns_only_invalidated():
    """Test that get_invalidated_memories returns only invalidated memories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = MemoryStorage(f"{tmpdir}/test.db")
        # Create memories
        memory1 = MemoryItem(content="Valid memory")
        memory2 = MemoryItem(content="Invalidated memory")
        memory3 = MemoryItem(content="Another valid memory")
        storage.save_memory(memory1)
        storage.save_memory(memory2)
        storage.save_memory(memory3)

        # Invalidate only memory2
        storage.invalidate(memory2.memory_id, reason="Test")

        # get_invalidated_memories should return only memory2
        invalidated = storage.get_invalidated_memories()
        assert len(invalidated) == 1
        assert invalidated[0].memory_id == memory2.memory_id


def test_is_invalidated_returns_correct_status():
    """Test that is_invalidated returns correct status."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = MemoryStorage(f"{tmpdir}/test.db")
        memory = MemoryItem(content="Test")
        storage.save_memory(memory)

        # Initially not invalidated
        assert storage.is_invalidated(memory.memory_id) is False

        # Invalidate
        storage.invalidate(memory.memory_id, reason="Test")

        # Now should be invalidated
        assert storage.is_invalidated(memory.memory_id) is True


def test_invalidation_preserves_record():
    """Test that invalidation preserves the original record (logical delete)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = MemoryStorage(f"{tmpdir}/test.db")
        memory = MemoryItem(
            content=f"Important content {tmpdir}",  # Unique per test
            tags=["important"],
            metadata={"key": "value"},
        )
        storage.save_memory(memory)
        original_id = memory.memory_id

        # Invalidate
        storage.invalidate(original_id, reason="Preserved for audit")

        # Record should still exist and be retrievable by admin
        retrieved = storage.get_memory(original_id, include_invalidated=True)
        assert retrieved is not None
        assert retrieved.memory_id == original_id
        assert retrieved.content == memory.content
        assert retrieved.tags == ["important"]
        assert retrieved.metadata == {"key": "value"}

        # Invalidation metadata should be present
        assert retrieved.invalidated_at is not None
        assert retrieved.invalidation_reason == "Preserved for audit"


def test_invalidation_timestamp_and_reason():
    """Test that invalidation records timestamp and reason."""
    from time import time

    with tempfile.TemporaryDirectory() as tmpdir:
        storage = MemoryStorage(f"{tmpdir}/test.db")
        memory = MemoryItem(content=f"Test content {tmpdir}")  # Unique per test
        storage.save_memory(memory)

        # Invalidate with specific reason
        before = time()
        storage.invalidate(memory.memory_id, reason="Duplicate entry")
        after = time()

        # Verify timestamp and reason
        retrieved = storage.get_memory(memory.memory_id, include_invalidated=True)
        assert retrieved.invalidated_at is not None
        assert before <= retrieved.invalidated_at <= after
        assert retrieved.invalidation_reason == "Duplicate entry"


def test_engine_retrieval_excludes_invalidated():
    """Test that FluidMemoryEngine retrieval excludes invalidated memories."""
    import uuid
    test_id = str(uuid.uuid4())[:8]

    with tempfile.TemporaryDirectory() as tmpdir:
        engine = FluidMemoryEngine(FluidMemoryConfig(db_path=f"{tmpdir}/engine.db"))

        # Add a memory
        memory = engine.add_memory(content=f"Test memory {test_id}")

        # Verify we can retrieve it
        results = engine.retrieve(query="memory", limit=10)
        assert len(results) >= 1
        assert any(r.memory.memory_id == memory.memory_id for r in results)

        # Invalidate via storage
        engine.storage.invalidate(memory.memory_id, reason="Test")

        # Search should not find invalidated memory
        results = engine.retrieve(query="memory", limit=10)
        assert not any(r.memory.memory_id == memory.memory_id for r in results)

        # Direct get should raise MemoryNotFoundError
        from fluid_memory.exceptions import MemoryNotFoundError
        with pytest.raises(MemoryNotFoundError):
            engine.get_memory(memory.memory_id)

        # But storage layer with include_invalidated can find it
        retrieved = engine.storage.get_memory(memory.memory_id, include_invalidated=True)
        assert retrieved is not None
        assert retrieved.invalidated_at is not None


def test_invalidation_idempotent():
    """Test that invalidating an already invalidated memory is safe."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = MemoryStorage(f"{tmpdir}/test.db")
        memory = MemoryItem(content=f"Test {tmpdir}")  # Unique per test
        storage.save_memory(memory)

        # First invalidation
        result1 = storage.invalidate(memory.memory_id, reason="First")
        assert result1 is True

        # Second invalidation should still succeed
        result2 = storage.invalidate(memory.memory_id, reason="Second")
        assert result2 is True

        # Should still be invalidated
        assert storage.is_invalidated(memory.memory_id) is True
