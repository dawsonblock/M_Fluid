# Fluid Memory

Standalone adaptive memory engine. Memories shift over time based on activation, decay, confidence, salience, and contradictions.

## Overview

Fluid Memory provides a dynamic memory field where each memory item tracks:
- **Salience**: How important the memory currently is
- **Confidence**: How reliable the memory appears
- **Volatility**: How likely the memory is to change
- **Decay rate**: How quickly salience decreases when unused

## Installation

Requires only:
- Python 3.8+
- Pydantic

```bash
pip install pydantic
```

## Quick Start

```python
from fluid_memory import FluidMemoryEngine, FluidMemoryConfig

# Create engine
config = FluidMemoryConfig(data_dir="./fluid_memory_data")
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

## Memory Model

Each memory tracks fluid state properties:

```python
memory_id: str              # UUID4 identifier
content: str                # Text content
content_hash: str           # MD5 of content for deduplication; persisted hashes are MD5-based, so changing algorithms later requires recomputing/migrating existing values for compatibility

salience: float             # [0.0-1.0] Current importance
confidence: float           # [0.0-1.0] Current reliability
volatility: float           # [0.0-1.0] Likelihood to change
stability: float            # [0.0-1.0] Resistance to decay
decay_rate: float           # [0.0-1.0] Salience decay per day

# Lane-specific salience
legal_salience: float       # Legal authority (slow decay)
trust_salience: float       # Trustworthiness
interest_salience: float    # Interestingness (fast decay)
attention_salience: float   # Attention (very fast decay)

reinforcement_count: int    # Confirming events
contradiction_count: int    # Conflicting events
```

## Decay Lanes

Decay is lane-specific:

| Lane | Default Rate | Use Case |
|------|--------------|----------|
| time | 0.05/day | General salience decay |
| legal | 0.02/day | Legal authority (slower) |
| trust | 0.03/day | Trustworthiness |
| interest | 0.08/day | Interestingness (faster) |
| attention | 0.15/day | Attention (very fast) |

## Operations

### Add Memory
```python
memory = engine.add_memory(
    content="Important fact",
    tags=["category"],
    source_refs=["source_id"],
    salience=0.8,
    confidence=0.9
)
```

### Retrieve
```python
results = engine.retrieve("query", limit=10)
# Results sorted by fluid score (salience * confidence)
```

### Reinforce
```python
engine.reinforce(memory_id, amount=0.1)
# Increases salience and confidence
```

### Contradict
```python
engine.contradict(memory_id, penalty=0.2)
# Decreases confidence
```

### Apply Decay
```python
count = engine.apply_decay(days=1.0)
# Applies decay to all memories
```

## Configuration

```python
FluidMemoryConfig(
    data_dir="./data",              # Storage directory
    default_decay_rate=0.05,        # General decay
    legal_decay_rate=0.02,          # Legal lane (slower)
    trust_decay_rate=0.03,          # Trust lane
    interest_decay_rate=0.08,       # Interest lane
    attention_decay_rate=0.15,      # Attention lane
    reinforcement_boost=0.1,          # Reinforcement amount
    contradiction_penalty=0.2,      # Contradiction penalty
    retrieval_threshold=0.1,        # Min retrieval score
    max_results=10,                 # Max results
)
```

## Deduplication

Memories are deduplicated by SHA256 content hash. Adding identical content does not create a new memory; `FluidMemoryEngine.add_memory()` raises `DuplicateMemoryError` when the same content hash already exists.

## Storage

- SQLite database for memory items
- JSON serialization for lists/dicts
- Automatic directory creation

## Status

Local prototype - testable but not production-grade.

See `docs/MEMORY_SYSTEM_STATUS.md` for full system architecture.
