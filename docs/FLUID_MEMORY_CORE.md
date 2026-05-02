# Fluid Memory Core

> **Status Note:** There are two fluid memory implementations:
> 1. `fluid_memory/` - Standalone package (this document)
> 2. `m_flow/memory/fluid/` - Advanced integrated engine (part of M-flow)
> 
> See `docs/MEMORY_SYSTEM_STATUS.md` for full architecture overview.

A standalone adaptive memory state engine. Memories are not static — they shift over time when touched by new input, reuse, contradictions, confirmations, age, and importance.

## Overview

Fluid Memory provides a dynamic memory field where each memory item tracks:
- **Salience**: How important the memory currently is
- **Confidence**: How reliable the memory appears
- **Volatility**: How likely the memory is to change
- **Stability**: How resistant to decay or mutation
- **Decay rate**: How quickly salience decreases when unused

## Installation

```bash
# Standalone - no dependencies except Pydantic
pip install pydantic

# Or with your project
from fluid_memory import FluidMemoryEngine, FluidMemoryConfig
```

## Quick Start

```python
from fluid_memory import FluidMemoryEngine, FluidMemoryConfig

# Create engine
config = FluidMemoryConfig(data_dir="/path/to/data")
engine = FluidMemoryEngine(config)

# Add memories
memory = engine.add_memory(
    "The sky is blue",
    tags=["facts", "science"],
    source_refs=["wikipedia_2024"]
)

# Retrieve with fluid scoring
results = engine.retrieve(query="sky", limit=5)
for result in results:
    print(f"{result.memory.content} (score: {result.score:.2f})")

# Reinforce important memories
engine.reinforce(memory.memory_id, amount=0.2)

# Apply decay to old unused memories
engine.apply_decay()

# Close when done
engine.close()
```

## Memory Item Structure

Each memory tracks:

```python
memory_id: str           # Unique identifier
content: str            # Text content
content_hash: str       # MD5 hash for duplicate detection
created_at: float       # Unix timestamp
updated_at: float       # Last modification timestamp
last_accessed_at: float # Last access timestamp
access_count: int       # Number of times retrieved

salience: float         # [0.0-1.0] Current importance
confidence: float       # [0.0-1.0] Current reliability
volatility: float       # [0.0-1.0] Likelihood to change
stability: float        # [0.0-1.0] Resistance to decay
decay_rate: float       # [0.0-1.0] Salience decay per day

reinforcement_count: int    # Confirming events
contradiction_count: int    # Conflicting events
source_refs: list[str]     # Source references
tags: list[str]            # Categorization tags
links: list[str]           # Linked memory IDs
metadata: dict             # Arbitrary metadata
```

## State Concepts

### Salience

How important the memory currently is. Factors affecting salience:
- **Access**: Each retrieval increases salience slightly
- **Reinforcement**: Confirming events boost salience significantly
- **Contradiction**: Conflicts may increase salience (attention-grabbing)

### Confidence

How reliable the memory appears:
- **Reinforcement**: Increases confidence
- **Contradiction**: Decreases confidence

### Volatility

How likely the memory is to change:
- **Contradiction**: Increases volatility
- **Stability**: Reduces volatility

High volatility = easier to update, lower resistance to mutation.

### Stability

How resistant the memory is to decay and mutation:
- **Reinforcement**: Increases stability
- **Contradiction**: Decreases stability

High stability = slower decay, more resistance to state changes.

### Decay

Memories lose salience when unused:

```
decay_amount = elapsed_days * decay_rate * (1.0 - stability)
new_salience = salience - decay_amount
```

- **Elapsed time**: Days since last access or update
- **Decay rate**: Configurable per-memory (default 0.05)
- **Stability**: Reduces decay rate proportionally

Decay never:
- Reduces salience below 0.0
- Modifies content
- Affects confidence directly

## Operations

### Reinforcement

Strengthen a memory through confirming events:

```python
engine.reinforce(
    memory_id,
    amount=0.1,              # Boost amount
    source_ref="doc_123",    # Optional source
    metadata={"reason": "confirmed"}
)
```

Effects:
- Increases salience
- Increases confidence
- Increases stability
- Decreases volatility
- Writes `reinforced` event

### Contradiction

Record a conflicting event:

```python
engine.contradict(
    memory_id,
    amount=0.1,
    source_ref="doc_456",
    metadata={"conflict": "new evidence"}
)
```

Effects:
- Decreases confidence
- Increases volatility
- Decreases stability slightly
- Increases salience slightly (attention)
- Writes `contradicted` event

### Mutation

Controlled update to memory content or state:

```python
engine.mutate(
    memory_id,
    new_content="Updated text",           # Optional content change
    state_delta={"salience": 0.9},         # Optional state changes
    reason="Correction",
    metadata={"author": "user"}
)
```

Rules:
- Content changes update content_hash
- State changes respect stability/volatility (mutation resistance)
- Writes `mutated` event with old/new values
- Does not erase source_refs, tags, or links

### Decay

Apply time-based salience reduction:

```python
decayed_count = engine.apply_decay(
    now=None,      # Optional timestamp override
    limit=None     # Optional max memories to process
)
```

Decay formula:
- Only affects memories with time since last access
- Writes `decayed` event only when values change
- Respects stability (stable memories decay slower)

## Retrieval

Retrieval combines multiple signals for ranking:

```python
results = engine.retrieve(
    query="search text",      # Text match
    tags=["important"],       # Tag filter
    limit=10
)
```

Scoring factors:
- **Text match**: Query appears in content
- **Tag match**: Matching tags present
- **Salience**: Higher salience = better rank
- **Confidence**: Higher confidence = better rank
- **Access count**: More accessed = higher rank

Results are `RetrievalResult` objects:

```python
result.memory      # MemoryItem
result.score       # [0.0-1.0] retrieval score
result.match_type  # "text", "tag", "combined", or "salience"
```

## Memory Links

Connect memories with typed relationships:

```python
engine.link_memories(
    source_id=memory1.memory_id,
    target_id=memory2.memory_id,
    link_type="supports",    # or "contradicts", "related", "parent", "child", "sequence"
    strength=0.8,
    metadata={"evidence": "strong"}
)
```

Link types:
- `related`: Generic connection
- `supports`: Source supports target
- `contradicts`: Source contradicts target
- `parent`: Source is parent of target
- `child`: Source is child of target
- `sequence`: Source comes before target

## Events

Every state change writes an event:

```python
events = engine.get_events(memory_id)
```

Event types:
- `created`: Memory created
- `accessed`: Memory retrieved
- `reinforced`: Memory reinforced
- `contradicted`: Memory contradicted
- `decayed`: Salience decayed
- `mutated`: Content or state mutated
- `linked`: Memory linked to another
- `deleted`: Memory deleted

Events include:
- `event_id`: Unique identifier
- `memory_id`: Affected memory
- `event_type`: Type of change
- `timestamp`: When it occurred
- `delta_json`: What changed (old/new values)
- `metadata_json`: Additional context

## Configuration

```python
from fluid_memory import FluidMemoryConfig

config = FluidMemoryConfig(
    data_dir="/path/to/data",           # Storage directory
    sqlite_path=None,                    # Or specific DB path
    default_salience=0.5,               # Initial salience
    default_confidence=0.5,             # Initial confidence
    default_volatility=0.3,             # Initial volatility
    default_stability=0.5,              # Initial stability
    default_decay_rate=0.05,            # Initial decay rate
    access_salience_boost=0.02,         # Access boost amount
    reinforcement_boost=0.1,            # Reinforcement boost
    contradiction_penalty=0.1,          # Contradiction penalty
    mutation_resistance_enabled=True,   # Enable resistance
)
```

## Storage

Uses SQLite with three tables:

- **memories**: All memory fields, JSON for lists/dicts
- **memory_events**: Event log with JSON deltas
- **memory_links**: Bidirectional memory relationships

Storage is persistent across engine restarts. Temporary directories work for testing.

## Architecture

```
┌─────────────────┐
│  Engine API     │  ← add_memory(), retrieve(), reinforce()
├─────────────────┤
│  Scoring        │  ← compute_retrieval_score()
├─────────────────┤
│  State Logic    │  ← decay, mutation, reinforcement
├─────────────────┤
│  Event System   │  ← record all changes
├─────────────────┤
│  SQLite Store   │  ← persistence
└─────────────────┘
```

No dependencies on:
- Graph databases
- Vector databases
- LLM APIs
- External logging frameworks

## Testing

Run tests with pytest:

```bash
pytest tests/test_fluid_imports.py -q
pytest tests/test_fluid_state.py -q
pytest tests/test_fluid_storage.py -q
pytest tests/test_fluid_scoring.py -q
pytest tests/test_fluid_retrieval.py -q
pytest tests/test_fluid_decay.py -q
pytest tests/test_fluid_mutation.py -q
pytest tests/test_fluid_events.py -q
```

Or all at once:

```bash
pytest tests/test_fluid_*.py -q
```

## Limitations

This implementation:
- Uses MD5 for content hashing (not cryptographically secure)
- Uses simple text matching for retrieval (no semantic search)
- Stores events indefinitely (no rotation policy)
- Uses SQLite (not distributed)
- Is single-process (no concurrent access handling)

## License

Same as M-Flow project.
