"""
Fluid Memory Mutation

Controlled mutation of memory content and state.
"""

import hashlib
import uuid
from typing import Optional, Dict, Any, Set
from time import time

from fluid_memory.models import MemoryItem
from fluid_memory.state import clamp01
from fluid_memory.events import MemoryEvent, EventType


# Fields that CAN be mutated via state_delta
# Protected fields (identity, integrity, audit) are blocked
MUTATION_ALLOWLIST: Set[str] = {
    # Fluid state fields (mutable)
    "salience",
    "confidence",
    "volatility",
    "stability",
    "decay_rate",
    "legal_salience",
    "trust_salience",
    "interest_salience",
    "attention_salience",
    # Reference fields (append-only via special handling)
    "source_refs",
    "tags",
    "links",
    # Metadata (controlled)
    "metadata",
}

# Fields that are PROTECTED and can NEVER be mutated directly
PROTECTED_FIELDS: Set[str] = {
    "memory_id",           # Identity - immutable
    "content_hash",        # Integrity - computed only
    "created_at",          # Audit - set at creation
    "updated_at",          # Audit - controlled internally
    "last_accessed_at",    # Audit - controlled internally
    "access_count",        # Audit - controlled via touch()
    "reinforcement_count", # Audit - controlled via reinforce()
    "contradiction_count", # Audit - controlled via contradict()
    "state_checksum",      # Integrity - computed only
    "invalidated_at",      # Audit - controlled via invalidate()
    "invalidation_reason", # Audit - controlled via invalidate()
    "content",             # Content - requires special handling with dedup check
}


def compute_mutation_resistance(
    stability: float,
    volatility: float,
    resistance_enabled: bool = True,
) -> float:
    """
    Compute how much a memory resists mutation.
    
    Higher stability = more resistance.
    Higher volatility = less resistance.
    
    Args:
        stability: Memory stability
        volatility: Memory volatility
        resistance_enabled: Whether resistance is active
        
    Returns:
        Resistance factor in [0.0, 1.0]
    """
    if not resistance_enabled:
        return 0.0
    
    # Stability increases resistance, volatility decreases it
    resistance = stability * (1.0 - volatility)
    return clamp01(resistance)


def apply_state_delta(
    memory: MemoryItem,
    state_delta: Dict[str, Any],
    resistance: float = 0.0,
    strict_allowlist: bool = True,
) -> Dict[str, Any]:
    """
    Apply a state delta to memory, respecting resistance and allowlist.

    Args:
        memory: Memory to mutate
        state_delta: Dict of field changes
        resistance: Mutation resistance factor
        strict_allowlist: If True, reject fields not in MUTATION_ALLOWLIST

    Returns:
        Applied changes (may be reduced due to resistance)

    Raises:
        ValueError: If strict_allowlist=True and protected field is attempted
    """
    applied = {}

    for field, new_value in state_delta.items():
        # Check field exists
        if not hasattr(memory, field):
            continue

        # Enforce allowlist
        if strict_allowlist and field not in MUTATION_ALLOWLIST:
            if field in PROTECTED_FIELDS:
                raise ValueError(
                    f"Cannot mutate protected field '{field}'. "
                    f"Use appropriate method instead."
                )
            # Field not in allowlist but not protected - skip silently
            continue

        old_value = getattr(memory, field)

        # For float fields, apply resistance to the change
        if isinstance(new_value, (int, float)) and isinstance(old_value, float):
            change = float(new_value) - old_value
            resisted_change = change * (1.0 - resistance)
            final_value = old_value + resisted_change
            setattr(memory, field, clamp01(final_value))
            applied[field] = {"old": old_value, "new": clamp01(final_value)}
        # For int fields
        elif isinstance(new_value, int) and isinstance(old_value, int):
            change = new_value - old_value
            resisted_change = int(change * (1.0 - resistance))
            final_value = old_value + resisted_change
            setattr(memory, field, final_value)
            applied[field] = {"old": old_value, "new": final_value}
        # For list fields (append only)
        elif isinstance(new_value, list) and isinstance(old_value, list):
            # Add new items
            existing_set = set(old_value)
            new_items = [item for item in new_value if item not in existing_set]
            if new_items:
                setattr(memory, field, old_value + new_items)
                applied[field] = {"added": new_items}
        # For dict fields (metadata) - merge, don't replace
        elif isinstance(new_value, dict) and isinstance(old_value, dict):
            merged = {**old_value, **new_value}
            setattr(memory, field, merged)
            applied[field] = {"merged_keys": list(new_value.keys())}
        # For other fields, replace if resistance is low
        else:
            if resistance < 0.5:  # Only replace if resistance < 50%
                setattr(memory, field, new_value)
                applied[field] = {"old": old_value, "new": new_value}

    return applied


def _compute_content_hash(content: str) -> str:
    """Compute SHA256 hash of content through controlled path.

    This ensures all content mutations use the same hashing method.
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class DuplicateContentError(ValueError):
    """Raised when mutation would create duplicate content."""
    pass


def mutate_memory(
    memory: MemoryItem,
    new_content: Optional[str] = None,
    state_delta: Optional[Dict[str, Any]] = None,
    reason: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    mutation_resistance_enabled: bool = True,
    allow_duplicate_content: bool = False,
    existing_hashes: Optional[set] = None,
) -> tuple[MemoryItem, MemoryEvent]:
    """
    Perform controlled mutation on a memory.

    Args:
        memory: Memory to mutate
        new_content: New content (if changing content)
        state_delta: State field changes
        reason: Reason for mutation
        metadata: Additional metadata
        mutation_resistance_enabled: Whether to apply resistance
        allow_duplicate_content: If False, raises DuplicateContentError
        existing_hashes: Set of existing content hashes to check against

    Returns:
        Tuple of (updated memory, mutation event)

    Raises:
        DuplicateContentError: If content would duplicate existing and allow_duplicate_content=False
        ValueError: If attempting to mutate protected fields via state_delta
    """
    now = time()
    old_hash = memory.content_hash
    applied_changes = {}

    # Apply content change if provided
    if new_content is not None and new_content != memory.content:
        # Compute new hash through controlled path
        new_hash = _compute_content_hash(new_content)

        # Check for duplicate content
        if not allow_duplicate_content and existing_hashes and new_hash in existing_hashes:
            raise DuplicateContentError(
                f"Content mutation rejected: would duplicate existing memory with hash {new_hash[:16]}..."
            )

        memory.content = new_content
        memory.content_hash = new_hash
        applied_changes["content"] = {
            "old_hash": old_hash,
            "new_hash": memory.content_hash,
        }

    # Apply state delta with strict allowlist
    if state_delta:
        resistance = compute_mutation_resistance(
            memory.stability,
            memory.volatility,
            mutation_resistance_enabled,
        )
        # strict_allowlist=True enforces field protection
        state_changes = apply_state_delta(memory, state_delta, resistance, strict_allowlist=True)
        if state_changes:
            applied_changes["state"] = state_changes

    # Update timestamp
    memory.touch()

    # Create event with UUID4 (not Python hash)
    event = MemoryEvent(
        event_id=f"evt_{uuid.uuid4().hex[:20]}",
        memory_id=memory.memory_id,
        event_type=EventType.MUTATED,
        timestamp=now,
        delta_json=applied_changes,
        metadata_json={
            "reason": reason,
            **(metadata or {}),
        },
    )
    
    return memory, event
