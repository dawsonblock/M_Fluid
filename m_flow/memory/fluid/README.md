# Fluid Memory Module

A mutable operational state layer for M-flow that creates a "water effect" in memory: when one source touches a node, nearby memory ripples with activation changes.

## Core Concept

**Raw evidence never changes.** Graph links change carefully. **Fluid state changes constantly.** Summaries can be regenerated from evidence + current state.

## Architecture

```
Episodic Memory Write → Fluid Engine Touch → Retrieval Scoring
                         (activation, decay,       (fluid_score
                          propagation)              boost/penalty)
```

## Quick Start

```python
from m_flow.memory.fluid import (
    FluidMemoryEngine,
    FluidStateStore,
    FluidUpdateEvent,
    fluid_score,
    get_source_weights,
)

# Create engine
store = FluidStateStore(db_provider="sqlite", db_path="/path/to/db")
engine = FluidMemoryEngine(graph_engine, store)

# After episodic memory write
await engine.touch(FluidUpdateEvent(
    touched_node_ids=[episode.id, facet.id, entity.id],
    source_id=document.id,
    source_type="mainstream_news",  # See source weights below
    source_trust=0.60,
    salience=0.7,
    legal_weight=0.3,
))

# In retrieval scoring
fluid_state = await engine.get_state(episode_id)
adjusted_score = fluid_score(base_retrieval_score, fluid_state)
```

## Source Trust Weights (Legal/Crime Use Case)

| Source | source_trust | legal_weight |
|--------|--------------|--------------|
| court_record | 0.95 | 1.00 |
| police_release | 0.80 | 0.70 |
| government_data | 0.85 | 0.80 |
| mainstream_news | 0.60 | 0.30 |
| blog_social | 0.25 | 0.05 |
| unknown | 0.10 | 0.00 |

```python
trust, legal = get_source_weights("court_record")
# trust = 0.95, legal = 1.00
```

## Scoring Weights

The `fluid_score()` function combines base retrieval score with fluid state:

| Factor | Weight | Effect |
|--------|--------|--------|
| activation | 0.25 | Recently touched nodes get boost |
| confidence | 0.25 | High-confidence nodes get boost |
| source_trust | 0.15 | Trusted sources boost their nodes |
| recency_score | 0.10 | Fresh nodes get boost |
| salience | 0.10 | Salient nodes get boost |
| legal_weight | 0.10 | Legal docs boost their nodes |
| contradiction_pressure | -0.20 | Contradicted nodes get penalty |

## Module Structure

| File | Purpose |
|------|---------|
| `models.py` | Pydantic models: `FluidMemoryState`, `FluidUpdateEvent` |
| `state_store.py` | SQLAlchemy storage for fluid state |
| `engine.py` | `FluidMemoryEngine` - main orchestrator |
| `scoring.py` | `fluid_score()` and boost computation |
| `decay.py` | Temporal decay of activation |
| `contradiction.py` | Pressure from conflicting sources |
| `propagation.py` | Activation ripple through graph (depth-2 BFS) |
| `audit.py` | Audit logging for all state mutations |
| `integration_example.py` | Shows how to hook into M-flow |

## FluidMemoryState Fields

- `node_id`: Episode/Facet/Entity/Point identifier
- `activation`: Current activation [0-1] - decays over time
- `confidence`: Source-derived confidence [0-1]
- `source_trust`: Max trust of sources that touched this node [0-1]
- `recency_score`: Time-based freshness [0-1]
- `decay_rate`: Per-node decay constant (default 0.01)
- `reinforcement_count`: Times node has been touched
- `contradiction_pressure`: Pressure from conflicting claims [0-1]
- `salience`: User/system flagged importance [0-1]
- `legal_weight`: Legal/court document weight [0-1]
- `last_touched_at`: Unix timestamp

## Integration Points

### 1. After Episodic Write

In `m_flow/memory/episodic/write_episodic_memories.py`, after episode nodes are created:

```python
from m_flow.memory.fluid import FluidMemoryEngine, FluidUpdateEvent, get_source_weights

# Collect node IDs
touched_ids = [episode.id]
touched_ids.extend(f.id for _, f in (episode.has_facet or []))
touched_ids.extend(e.id for _, e in (episode.involves_entity or []))

# Get source weights
trust, legal = get_source_weights(document.source_type)

# Touch fluid memory
await fluid_engine.touch(FluidUpdateEvent(
    touched_node_ids=touched_ids,
    source_id=document.id,
    source_type=document.source_type,
    source_trust=trust,
    salience=0.7,
    legal_weight=legal,
))
```

### 2. In Retrieval Scoring

In `m_flow/retrieval/episodic/bundle_scorer.py`, in `compute_episode_bundles`:

```python
from m_flow.memory.fluid import fluid_score

# After computing base bundle score
for bundle in bundles:
    fluid_state = await fluid_store.get(bundle.episode_id)
    if fluid_state:
        bundle.score = fluid_score(bundle.score, fluid_state)
```

## How Activation Propagation Works

When a node is touched:

1. **Direct activation**: Touched node gets +0.25 activation (capped at 1.0)
2. **Depth-1 ripple**: Connected nodes get 0.18 * 0.5 = 0.09 activation
3. **Depth-2 ripple**: Nodes 2 hops away get 0.18 * 0.5 * 0.5 = 0.045 activation

Propagation follows edges: `has_facet`, `has_point`, `involves_entity`, `supported_by`, `includes_chunk`, `same_entity_as`.

## Decay

Activation decays exponentially: `activation * exp(-decay_rate * time_elapsed)`

Default decay_rate is 0.01, meaning ~1% decay per second of elapsed time.

## Testing

```bash
cd /Users/dawsonblock/Downloads/m_flow-main
PYTHONPATH=. uv run python -m pytest m_flow/tests/unit/memory/fluid/test_fluid_memory.py -v
```

## v0.1 Features

- [x] Node state table (SQLAlchemy/SQLite)
- [x] Touch/update function
- [x] Activation ripple through graph depth 2
- [x] Decay function
- [x] Contradiction pressure field
- [x] Retrieval score boost
- [x] Audit log
