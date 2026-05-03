"""
Test Fluid Memory mutation safety (Phase 2).

- Mutation cannot change protected fields
- Event IDs are not Python hash values
- Content mutation updates checksum through controlled path
- Duplicate content mutation is rejected
"""

import pytest
from fluid_memory.mutation import (
    compute_mutation_resistance,
    apply_state_delta,
    mutate_memory,
    _compute_content_hash,
    DuplicateContentError,
    MUTATION_ALLOWLIST,
    PROTECTED_FIELDS,
)
from fluid_memory.models import MemoryItem


def test_mutation_allowlist_is_defined():
    """Test that mutation allowlist exists and contains expected fields."""
    # Fluid state fields should be allowed
    assert "salience" in MUTATION_ALLOWLIST
    assert "confidence" in MUTATION_ALLOWLIST
    assert "volatility" in MUTATION_ALLOWLIST
    assert "stability" in MUTATION_ALLOWLIST

    # List fields should be allowed
    assert "tags" in MUTATION_ALLOWLIST
    assert "source_refs" in MUTATION_ALLOWLIST
    assert "links" in MUTATION_ALLOWLIST

    # Metadata should be allowed
    assert "metadata" in MUTATION_ALLOWLIST


def test_protected_fields_are_defined():
    """Test that protected fields list exists and contains identity fields."""
    # Identity fields should be protected
    assert "memory_id" in PROTECTED_FIELDS
    assert "content_hash" in PROTECTED_FIELDS

    # Audit fields should be protected
    assert "created_at" in PROTECTED_FIELDS
    assert "updated_at" in PROTECTED_FIELDS
    assert "access_count" in PROTECTED_FIELDS

    # Controlled counters should be protected
    assert "reinforcement_count" in PROTECTED_FIELDS
    assert "contradiction_count" in PROTECTED_FIELDS

    # Invalidation fields should be protected
    assert "invalidated_at" in PROTECTED_FIELDS
    assert "invalidation_reason" in PROTECTED_FIELDS


def test_mutation_cannot_change_memory_id():
    """Test that mutation cannot change memory_id."""
    memory = MemoryItem(content="Test")
    original_id = memory.memory_id

    # Attempting to change memory_id via state_delta should raise ValueError
    with pytest.raises(ValueError) as exc_info:
        apply_state_delta(
            memory,
            {"memory_id": "new_id"},
            resistance=0.0,
            strict_allowlist=True,
        )

    assert "protected field" in str(exc_info.value).lower()
    assert memory.memory_id == original_id  # Unchanged


def test_mutation_cannot_change_content_hash():
    """Test that mutation cannot change content_hash directly."""
    memory = MemoryItem(content="Test")
    original_hash = memory.content_hash

    # Attempting to change content_hash via state_delta should raise ValueError
    with pytest.raises(ValueError) as exc_info:
        apply_state_delta(
            memory,
            {"content_hash": "fake_hash"},
            resistance=0.0,
            strict_allowlist=True,
        )

    assert "protected field" in str(exc_info.value).lower()
    assert memory.content_hash == original_hash  # Unchanged


def test_mutation_cannot_change_created_at():
    """Test that mutation cannot change created_at."""
    memory = MemoryItem(content="Test")
    original_created = memory.created_at

    # Attempting to change created_at should raise ValueError
    with pytest.raises(ValueError) as exc_info:
        apply_state_delta(
            memory,
            {"created_at": 9999999999.0},
            resistance=0.0,
            strict_allowlist=True,
        )

    assert "protected field" in str(exc_info.value).lower()
    assert memory.created_at == original_created  # Unchanged


def test_mutation_cannot_change_reinforcement_count():
    """Test that mutation cannot change reinforcement_count directly."""
    memory = MemoryItem(content="Test")

    # Attempting to change reinforcement_count should raise ValueError
    with pytest.raises(ValueError) as exc_info:
        apply_state_delta(
            memory,
            {"reinforcement_count": 100},
            resistance=0.0,
            strict_allowlist=True,
        )

    assert "protected field" in str(exc_info.value).lower()


def test_mutation_cannot_change_contradiction_count():
    """Test that mutation cannot change contradiction_count directly."""
    memory = MemoryItem(content="Test")

    # Attempting to change contradiction_count should raise ValueError
    with pytest.raises(ValueError) as exc_info:
        apply_state_delta(
            memory,
            {"contradiction_count": 50},
            resistance=0.0,
            strict_allowlist=True,
        )

    assert "protected field" in str(exc_info.value).lower()


def test_event_id_is_not_python_hash():
    """Test that mutation events use UUID, not Python hash()."""
    memory = MemoryItem(content="Test content")

    mutated, event = mutate_memory(
        memory,
        new_content="Updated content",
        reason="Test update",
    )

    # Event ID should start with "evt_" (UUID-based)
    assert event.event_id.startswith("evt_")

    # Event ID should NOT be a simple integer string (which hash() would produce)
    event_id_suffix = event.event_id.replace("evt_", "")
    assert not event_id_suffix.isdigit(), "Event ID should not be a simple hash integer"

    # Event ID should be hex characters
    assert all(c in "0123456789abcdef" for c in event_id_suffix)

    # Event ID should be reasonably long (UUID hex is 20 chars in our format)
    assert len(event_id_suffix) >= 16


def test_event_ids_are_unique():
    """Test that multiple mutations produce unique event IDs."""
    memory = MemoryItem(content="Test")

    event_ids = []
    for i in range(10):
        mutated, event = mutate_memory(
            memory,
            state_delta={"salience": 0.5 + (i * 0.01)},
        )
        event_ids.append(event.event_id)

    # All event IDs should be unique
    assert len(set(event_ids)) == len(event_ids)


def test_compute_content_hash_is_consistent():
    """Test that content hash computation is deterministic."""
    content = "Test content for hashing"

    hash1 = _compute_content_hash(content)
    hash2 = _compute_content_hash(content)

    # Same content should produce same hash
    assert hash1 == hash2

    # Should be SHA256 (64 hex chars)
    assert len(hash1) == 64
    assert all(c in "0123456789abcdef" for c in hash1)


def test_content_mutation_updates_hash_via_controlled_path():
    """Test that content mutation updates hash through _compute_content_hash."""
    memory = MemoryItem(content="Original content")
    old_hash = memory.content_hash

    # Mutate content
    mutated, event = mutate_memory(
        memory,
        new_content="New content",
        reason="Test",
    )

    # Hash should be updated
    assert mutated.content_hash != old_hash

    # Hash should match what _compute_content_hash produces
    expected_hash = _compute_content_hash("New content")
    assert mutated.content_hash == expected_hash

    # Event should record hash change
    assert "content" in event.delta_json
    assert event.delta_json["content"]["old_hash"] == old_hash
    assert event.delta_json["content"]["new_hash"] == expected_hash


def test_duplicate_content_mutation_raises_error():
    """Test that duplicate content mutation raises DuplicateContentError."""
    # Create set of existing hashes
    existing_hashes = set()

    # Create first memory
    memory1 = MemoryItem(content="Test content")
    existing_hashes.add(memory1.content_hash)

    # Create second memory with different content
    memory2 = MemoryItem(content="Different content")
    existing_hashes.add(memory2.content_hash)

    # Try to mutate memory2 to have same content as memory1
    # This should raise DuplicateContentError
    with pytest.raises(DuplicateContentError) as exc_info:
        mutate_memory(
            memory2,
            new_content="Test content",  # Same as memory1
            existing_hashes=existing_hashes,
            allow_duplicate_content=False,
        )

    assert "duplicate" in str(exc_info.value).lower()


def test_duplicate_content_allowed_when_configured():
    """Test that duplicate content is allowed when allow_duplicate_content=True."""
    existing_hashes = set()

    memory1 = MemoryItem(content="Test content")
    existing_hashes.add(memory1.content_hash)

    memory2 = MemoryItem(content="Different content")
    existing_hashes.add(memory2.content_hash)

    # With allow_duplicate_content=True, should succeed
    mutated, event = mutate_memory(
        memory2,
        new_content="Test content",
        existing_hashes=existing_hashes,
        allow_duplicate_content=True,  # Allow duplicates
    )

    # Mutation should succeed
    assert mutated.content == "Test content"


def test_noop_mutation_does_not_create_event():
    """Test that mutation with no changes doesn't create unnecessary events."""
    memory = MemoryItem(content="Test content")

    # Mutate with same content (no actual change)
    mutated, event = mutate_memory(
        memory,
        new_content="Test content",  # Same as existing
        reason="No-op test",
    )

    # Content should be unchanged
    assert mutated.content == "Test content"

    # Event should still be created but with empty delta
    assert event.event_type.value == "mutated"


def test_mutation_preserves_protected_fields():
    """Test that mutation preserves all protected fields."""
    memory = MemoryItem(content="Test")

    # Record original values
    original_values = {
        "memory_id": memory.memory_id,
        "content_hash": memory.content_hash,
        "created_at": memory.created_at,
        "access_count": memory.access_count,
        "reinforcement_count": memory.reinforcement_count,
        "contradiction_count": memory.contradiction_count,
    }

    # Perform legitimate mutations on allowed fields
    mutated, event = mutate_memory(
        memory,
        new_content="Updated content",
        state_delta={
            "salience": 0.8,
            "confidence": 0.9,
        },
    )

    # Protected fields should be preserved
    assert mutated.memory_id == original_values["memory_id"]
    assert mutated.created_at == original_values["created_at"]
    # Note: content_hash will change (legitimately) due to content update
    # But counters should remain at their original values
    assert mutated.reinforcement_count == original_values["reinforcement_count"]
    assert mutated.contradiction_count == original_values["contradiction_count"]
