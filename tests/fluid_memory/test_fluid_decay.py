"""
Test Fluid Memory decay operations.
"""

import pytest
from time import time
from fluid_memory.decay import (
    compute_decay_amount,
    apply_decay_to_memory,
    apply_decay,
)
from fluid_memory.models import MemoryItem


def test_compute_decay_amount():
    """Test decay amount calculation."""
    amount = compute_decay_amount(
        salience=0.5,
        elapsed_days=10,
        decay_rate=0.05,
        stability=0.5,
    )
    
    # 10 days * 0.05 rate * (1 - 0.5 stability) = 0.25
    assert amount == 0.25


def test_compute_decay_amount_with_high_stability():
    """Test that high stability reduces decay."""
    low_stability = compute_decay_amount(
        salience=0.5,
        elapsed_days=10,
        decay_rate=0.05,
        stability=0.1,
    )
    
    high_stability = compute_decay_amount(
        salience=0.5,
        elapsed_days=10,
        decay_rate=0.05,
        stability=0.9,
    )
    
    assert high_stability < low_stability


def test_apply_decay_to_memory():
    """Test applying decay to a single memory."""
    memory = MemoryItem(content="Test")
    memory.salience = 0.8
    memory.updated_at = time() - 86400 * 5  # 5 days ago
    
    updated, was_decayed = apply_decay_to_memory(memory)
    
    assert was_decayed is True
    assert updated.salience < 0.8
    assert updated.salience >= 0.0


def test_apply_decay_to_fresh_memory():
    """Test that fresh memory doesn't decay."""
    memory = MemoryItem(content="Test")
    memory.salience = 0.8
    # updated_at is now by default
    
    updated, was_decayed = apply_decay_to_memory(memory)
    
    assert was_decayed is False
    assert updated.salience == 0.8


def test_apply_decay_respects_min_salience():
    """Test that decay respects minimum salience floor."""
    memory = MemoryItem(content="Test")
    memory.salience = 0.1
    memory.updated_at = time() - 86400 * 365  # 1 year ago
    
    updated, was_decayed = apply_decay_to_memory(memory, min_salience=0.05)
    
    assert updated.salience >= 0.05


def test_apply_decay_content_unchanged():
    """Test that content is never modified by decay."""
    memory = MemoryItem(content="Important content")
    memory.updated_at = time() - 86400 * 10
    
    updated, _ = apply_decay_to_memory(memory)
    
    assert updated.content == "Important content"
    assert updated.content_hash == memory.content_hash


def test_apply_decay_many_memories():
    """Test applying decay to multiple memories."""
    old_memory = MemoryItem(content="Old")
    old_memory.updated_at = time() - 86400 * 30
    old_memory.salience = 0.8
    
    new_memory = MemoryItem(content="New")
    # updated_at is now
    new_memory.salience = 0.8
    
    memories = [old_memory, new_memory]
    updated, events = apply_decay(memories)
    
    assert len(updated) == 2
    assert len(events) == 1  # Only old memory decayed
    assert events[0].event_type.value == "decayed"
    
    # Old memory should have decayed
    old_updated = next(m for m in updated if m.content == "Old")
    assert old_updated.salience < 0.8
    
    # New memory should not have decayed
    new_updated = next(m for m in updated if m.content == "New")
    assert new_updated.salience == 0.8


def test_apply_decay_with_limit():
    """Test that limit restricts number of processed memories."""
    memories = []
    for i in range(10):
        m = MemoryItem(content=f"Memory {i}")
        m.updated_at = time() - 86400 * 10
        memories.append(m)
    
    updated, events = apply_decay(memories, limit=3)
    
    assert len(updated) == 3
    assert len(events) <= 3
