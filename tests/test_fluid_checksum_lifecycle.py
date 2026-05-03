"""Test checksum lifecycle: automatic updates after mutations and verification."""

import tempfile
import pytest
from fluid_memory import FluidMemoryEngine, FluidMemoryConfig
from fluid_memory.models import MemoryItem


def test_checksum_auto_updated_after_add_memory():
    """Checksum is automatically computed after add_memory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = FluidMemoryEngine(FluidMemoryConfig(db_path=f"{tmpdir}/test.db"))
        
        memory = engine.add_memory(content="Test content", detect_contradictions=False)
        
        # Verify checksum was computed
        assert engine.verify_memory(memory.memory_id) is True


def test_checksum_auto_updated_after_get_memory():
    """Checksum remains valid after get_memory access touch."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = FluidMemoryEngine(FluidMemoryConfig(db_path=f"{tmpdir}/test.db"))
        memory = engine.add_memory(content="Test content", detect_contradictions=False)
        assert engine.verify_memory(memory.memory_id) is True
        engine.get_memory(memory.memory_id)
        assert engine.verify_memory(memory.memory_id) is True


def test_checksum_auto_updated_after_reinforce():
    """Checksum is automatically updated after reinforce."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = FluidMemoryEngine(FluidMemoryConfig(db_path=f"{tmpdir}/test.db"))
        
        memory = engine.add_memory(content="Test content", detect_contradictions=False)
        
        # Reinforce and verify checksum still valid
        engine.reinforce(memory.memory_id, amount=0.1)
        assert engine.verify_memory(memory.memory_id) is True


def test_checksum_auto_updated_after_contradict():
    """Checksum is automatically updated after contradict."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = FluidMemoryEngine(FluidMemoryConfig(db_path=f"{tmpdir}/test.db"))
        
        memory = engine.add_memory(content="Test content", detect_contradictions=False)
        
        # Contradict and verify checksum still valid
        engine.contradict(memory.memory_id, amount=0.1)
        assert engine.verify_memory(memory.memory_id) is True


def test_checksum_auto_updated_after_mutate():
    """Checksum is automatically updated after mutate."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = FluidMemoryEngine(FluidMemoryConfig(db_path=f"{tmpdir}/test.db"))
        
        memory = engine.add_memory(content="Test content", detect_contradictions=False)
        
        # Mutate and verify checksum still valid
        engine.mutate(memory.memory_id, new_content="Mutated content")
        assert engine.verify_memory(memory.memory_id) is True


def test_checksum_auto_updated_after_apply_decay():
    """Checksum is automatically updated after apply_decay."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = FluidMemoryEngine(FluidMemoryConfig(db_path=f"{tmpdir}/test.db"))
        
        memory = engine.add_memory(content="Test content", detect_contradictions=False)
        
        # Apply decay and verify checksum still valid
        engine.apply_decay(days=1.0)
        assert engine.verify_memory(memory.memory_id) is True


def test_engine_invalidate_memory_method_exists():
    """Engine has invalidate_memory method as convenience wrapper."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = FluidMemoryEngine(FluidMemoryConfig(db_path=f"{tmpdir}/test.db"))
        
        memory = engine.add_memory(content="Test content", detect_contradictions=False)
        
        # Use engine-level invalidate
        result = engine.invalidate_memory(memory.memory_id, reason="Test reason")
        
        assert result.invalidated_at is not None
        assert result.invalidation_reason == "Test reason"
        
        # Verify memory is now hidden from normal retrieval
        with pytest.raises(Exception):
            engine.get_memory(memory.memory_id)


def test_engine_verify_memory_method():
    """Engine-level verify_memory wraps storage method."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = FluidMemoryEngine(FluidMemoryConfig(db_path=f"{tmpdir}/test.db"))
        
        memory = engine.add_memory(content="Test content", detect_contradictions=False)
        
        # Verify via engine
        assert engine.verify_memory(memory.memory_id) is True


def test_engine_verify_all_memory_checksums():
    """Engine-level verify_all_memory_checksums wraps storage method."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = FluidMemoryEngine(FluidMemoryConfig(db_path=f"{tmpdir}/test.db"))
        
        # Add multiple memories
        m1 = engine.add_memory(content="Content 1", detect_contradictions=False)
        m2 = engine.add_memory(content="Content 2", detect_contradictions=False)
        m3 = engine.add_memory(content="Content 3", detect_contradictions=False)
        
        # Verify all
        result = engine.verify_all_memory_checksums()
        
        assert result["total"] == 3
        assert result["valid"] == 3
        assert result["invalid"] == 0
        assert len(result["errors"]) == 0


def test_verify_all_detects_corrupted_memory():
    """verify_all_memory_checksums detects manually corrupted state."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = FluidMemoryEngine(FluidMemoryConfig(db_path=f"{tmpdir}/test.db"))
        
        memory = engine.add_memory(content="Test content", detect_contradictions=False)
        
        # Manually corrupt a STATE field in storage (bypass engine)
        # Note: checksum is based on state fields (salience, confidence, etc.), not content
        raw_memory = engine.storage.get_memory(memory.memory_id, include_invalidated=True)
        raw_memory.salience = 0.99  # Corrupt the salience
        # Don't update checksum after corruption
        engine.storage.update_memory(raw_memory)
        
        # Verify all should detect the corruption
        result = engine.verify_all_memory_checksums()
        assert result["total"] == 1
        assert result["valid"] == 0
        assert result["invalid"] == 1
        assert memory.memory_id in [e["memory_id"] for e in result["errors"]]


def test_checksum_lifecycle_comprehensive():
    """Full lifecycle: create, mutate, reinforce, contradict, verify, invalidate."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = FluidMemoryEngine(FluidMemoryConfig(db_path=f"{tmpdir}/test.db"))
        
        # 1. Create
        memory = engine.add_memory(content="Initial", detect_contradictions=False)
        assert engine.verify_memory(memory.memory_id)
        
        # 2. Mutate
        engine.mutate(memory.memory_id, new_content="Mutated")
        assert engine.verify_memory(memory.memory_id)
        
        # 3. Reinforce
        engine.reinforce(memory.memory_id, amount=0.2)
        assert engine.verify_memory(memory.memory_id)
        
        # 4. Contradict
        engine.contradict(memory.memory_id, amount=0.1)
        assert engine.verify_memory(memory.memory_id)
        
        # 5. Apply decay
        engine.apply_decay(days=1.0)
        assert engine.verify_memory(memory.memory_id)
        
        # 6. Verify all still passing
        result = engine.verify_all_memory_checksums()
        assert result["valid"] == 1
        
        # 7. Invalidate
        engine.invalidate_memory(memory.memory_id, reason="Done testing")


def test_checksum_auto_updated_after_invalidate_memory():
    """Checksum remains valid after engine-level invalidation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = FluidMemoryEngine(FluidMemoryConfig(db_path=f"{tmpdir}/test.db"))
        memory = engine.add_memory(content="Test content", detect_contradictions=False)
        assert engine.verify_memory(memory.memory_id) is True
        engine.invalidate_memory(memory.memory_id, reason="No longer valid")
        assert engine.verify_memory(memory.memory_id) is True


def test_checksum_detects_volatility_corruption():
    """Checksum detects corruption of volatility field (newly covered field)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = FluidMemoryEngine(FluidMemoryConfig(db_path=f"{tmpdir}/test.db"))
        
        memory = engine.add_memory(content="Test content", detect_contradictions=False)
        
        # Verify initial checksum is valid
        assert engine.verify_memory(memory.memory_id) is True
        
        # Manually corrupt volatility field (bypass engine)
        raw_memory = engine.storage.get_memory(memory.memory_id, include_invalidated=True)
        raw_memory.volatility = 0.99  # Corrupt the volatility
        engine.storage.update_memory(raw_memory)
        
        # Verify checksum now fails (volatility is now covered)
        assert engine.verify_memory(memory.memory_id) is False
        
        # Verify all should also detect it
        result = engine.verify_all_memory_checksums()
        assert result["invalid"] == 1
        assert memory.memory_id in [e["memory_id"] for e in result["errors"]]
