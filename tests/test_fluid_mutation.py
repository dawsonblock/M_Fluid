"""
Test Fluid Memory mutation operations.
"""

import pytest
from fluid_memory.mutation import (
    compute_mutation_resistance,
    apply_state_delta,
    mutate_memory,
)
from fluid_memory.models import MemoryItem


def test_compute_mutation_resistance():
    """Test mutation resistance calculation."""
    # High stability, low volatility = high resistance
    high_resistance = compute_mutation_resistance(
        stability=0.9,
        volatility=0.1,
        resistance_enabled=True,
    )
    assert high_resistance > 0.5
    
    # Low stability, high volatility = low resistance
    low_resistance = compute_mutation_resistance(
        stability=0.1,
        volatility=0.9,
        resistance_enabled=True,
    )
    assert low_resistance < 0.5


def test_compute_mutation_resistance_disabled():
    """Test that resistance can be disabled."""
    resistance = compute_mutation_resistance(
        stability=0.9,
        volatility=0.1,
        resistance_enabled=False,
    )
    assert resistance == 0.0


def test_apply_state_delta_float_fields():
    """Test applying state delta to float fields."""
    memory = MemoryItem(content="Test")
    memory.salience = 0.5
    
    delta = {"salience": 0.8}
    applied = apply_state_delta(memory, delta, resistance=0.0)
    
    assert "salience" in applied
    assert memory.salience > 0.5


def test_apply_state_delta_with_resistance():
    """Test that resistance reduces state changes."""
    memory1 = MemoryItem(content="Test 1")
    memory1.salience = 0.5
    
    memory2 = MemoryItem(content="Test 2")
    memory2.salience = 0.5
    
    delta = {"salience": 0.9}
    
    # No resistance
    apply_state_delta(memory1, delta, resistance=0.0)
    
    # High resistance
    apply_state_delta(memory2, delta, resistance=0.8)
    
    # Memory with no resistance should change more
    assert memory1.salience > memory2.salience


def test_apply_state_delta_int_fields():
    """Test applying state delta to int fields."""
    memory = MemoryItem(content="Test")
    memory.access_count = 5
    
    delta = {"access_count": 10}
    applied = apply_state_delta(memory, delta, resistance=0.0)
    
    assert memory.access_count > 5


def test_apply_state_delta_list_fields():
    """Test applying state delta to list fields (append only)."""
    memory = MemoryItem(content="Test")
    memory.tags = ["existing"]
    
    delta = {"tags": ["new1", "new2"]}
    applied = apply_state_delta(memory, delta, resistance=0.0)
    
    assert "new1" in memory.tags
    assert "new2" in memory.tags
    assert "existing" in memory.tags


def test_mutate_memory_content():
    """Test mutating memory content."""
    memory = MemoryItem(content="Original content")
    old_hash = memory.content_hash
    
    mutated, event = mutate_memory(
        memory,
        new_content="New content",
        reason="Test update",
    )
    
    assert mutated.content == "New content"
    assert mutated.content_hash != old_hash
    assert event.event_type.value == "mutated"
    assert "content" in event.delta_json


def test_mutate_memory_state():
    """Test mutating memory state."""
    memory = MemoryItem(content="Test")
    memory.salience = 0.5
    
    mutated, event = mutate_memory(
        memory,
        state_delta={"salience": 0.9},
    )
    
    assert mutated.salience > 0.5
    assert "state" in event.delta_json


def test_mutate_memory_preserves_lists():
    """Test that mutation doesn't erase lists."""
    memory = MemoryItem(content="Test")
    memory.tags = ["tag1"]
    memory.source_refs = ["ref1"]
    memory.links = ["link1"]
    
    mutated, _ = mutate_memory(
        memory,
        new_content="Updated content",
    )
    
    assert "tag1" in mutated.tags
    assert "ref1" in mutated.source_refs
    assert "link1" in mutated.links


def test_mutate_memory_respects_volatility():
    """Test that volatile memories change more easily."""
    stable_memory = MemoryItem(content="Stable")
    stable_memory.stability = 0.9
    stable_memory.volatility = 0.1
    stable_memory.salience = 0.5
    
    volatile_memory = MemoryItem(content="Volatile")
    volatile_memory.stability = 0.1
    volatile_memory.volatility = 0.9
    volatile_memory.salience = 0.5
    
    delta = {"salience": 0.9}
    
    mutated_stable, _ = mutate_memory(
        stable_memory,
        state_delta=delta,
        mutation_resistance_enabled=True,
    )
    
    mutated_volatile, _ = mutate_memory(
        volatile_memory,
        state_delta=delta,
        mutation_resistance_enabled=True,
    )
    
    # Volatile memory should change more
    assert mutated_volatile.salience > mutated_stable.salience
