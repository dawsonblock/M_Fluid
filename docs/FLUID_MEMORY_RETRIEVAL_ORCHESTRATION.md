# Fluid Memory Retrieval and Orchestration

Testable retrieval/grounding layer for evidence-aware memory answers.

## What This Layer Does

This module adds evidence-grounded, provenance-aware, conflict-aware retrieval capabilities to fluid_memory. It provides:

1. **Retrieval Packets** — Structured containers for retrieval results with evidence references and support level assessment
2. **Conflict-Aware Reranking** — Penalizes contradicted/volatile memories without deleting them
3. **Answer Grounding** — Validates whether an answer should be given based on evidence quality

## What It Does Not Do

- This is **not** a truth oracle. It does not guarantee factual correctness.
- It does **not** provide legal, medical, or financial advice.
- It does **not** replace human judgment for critical decisions.
- Evidence references are provenance aids, not proof by themselves.

## Retrieval Packet

A `RetrievalPacket` contains:

- `query`: The original search query
- `results`: List of `RetrievalResult` objects
- `evidence_refs`: List of `MemoryEvidenceRef` objects with provenance info
- `support_level`: Assessment of evidence quality
- `warnings`: Any concerns about the evidence

### Support Levels

| Level | Meaning |
|-------|---------|
| `none` | No results found |
| `weak` | Results exist but scores are low or confidence is poor |
| `mixed` | Results have contradictions or high volatility |
| `supported` | Decent evidence exists but lacks strong corroboration |
| `strong` | Multiple high-confidence, stable, corroborating results |

### Usage

```python
from fluid_memory import FluidMemoryEngine, FluidMemoryConfig

engine = FluidMemoryEngine(FluidMemoryConfig(db_path="memory.db"))
orchestrator = engine.create_orchestrator()

# Retrieve with evidence grounding
packet = orchestrator.retrieve_packet(
    query="Python programming",
    use_semantic=True,
    conflict_aware=True,
)

print(f"Support level: {packet.support_level}")
print(f"Evidence count: {len(packet.evidence_refs)}")
for ref in packet.evidence_refs:
    print(f"  - {ref.memory_id}: confidence={ref.confidence}")
```

## Conflict-Aware Reranking

When `conflict_aware=True` (default), the orchestrator penalizes memories with:

- High contradiction counts
- High volatility
- Low stability

The penalty is applied as a multiplier: `adjusted_score = raw_score * (1 - penalty)`

Penalties are clamped to 0.0–0.8, so a memory can never be fully eliminated by conflict scoring alone.

### Example

```python
# Without conflict awareness
packet = orchestrator.retrieve_packet(query="Python", conflict_aware=False)

# With conflict awareness (default)
packet = orchestrator.retrieve_packet(query="Python", conflict_aware=True)
```

## Answer Grounding

The `ground_answer()` method determines whether an answer should be given based on the evidence packet:

```python
packet = orchestrator.retrieve_packet(query="What is Python?")

grounded = orchestrator.ground_answer(
    query="What is Python?",
    answer="Python is a programming language",
    packet=packet,
)

print(f"Should answer: {grounded['should_answer']}")
print(f"Support level: {grounded['support_level']}")
print(f"Warnings: {grounded['warnings']}")

# Evidence for the answer
for evidence in grounded['evidence']:
    print(f"  Source: {evidence['source_refs']}")
    print(f"  Confidence: {evidence['confidence']}")
```

### Grounding Rules

- `support_level="none"` → `should_answer=False`
- `support_level="supported"` or `"strong"` → `should_answer=True`
- `support_level="mixed"` → `should_answer=True` but with warnings
- `support_level="weak"` → `should_answer` depends on use case

## Evidence References

Each `MemoryEvidenceRef` includes:

- `memory_id`: Unique identifier
- `content_hash`: For integrity verification
- `source_refs`: Original source references
- `tags`: Memory categorization
- `confidence`: Reliability score
- `salience`: Importance score
- `stability`: Resistance to decay
- `volatility`: Change susceptibility
- `contradiction_count`: Number of contradictions recorded
- `invalidated`: Whether memory has been invalidated

## Known Limitations

1. **Semantic search** uses a lightweight hash-based embedding. For production use, replace with proper sentence transformers.

2. **Conflict scoring** is heuristic-based. It does not understand semantic contradiction—only counts recorded contradiction events.

3. **Evidence refs** are pointers to memory state. They do not capture the full context of when/why a memory was created.

4. **No temporal reasoning** — the system does not understand that newer memories might override older ones.

5. **SQLite only** — no connection pooling or horizontal scaling.

## Testing

Run the orchestration tests:

```bash
PYTHONPATH=. python -m pytest tests/test_fluid_retrieval_packets.py -v
PYTHONPATH=. python -m pytest tests/test_fluid_conflict_reranking.py -v
PYTHONPATH=. python -m pytest tests/test_fluid_orchestrator_grounding.py -v
PYTHONPATH=. python -m pytest tests/test_fluid_memory_benchmarks.py -v
```

## Architecture

```
FluidMemoryEngine
    ├── storage (SQLite)
    ├── health (basic checks)
    ├── metrics (manual collection)
    ├── batch (bulk operations)
    └── create_orchestrator() → MemoryOrchestrator

MemoryOrchestrator
    ├── retrieve_packet() → RetrievalPacket
    └── ground_answer() → GroundedAnswer

RetrievalPacket
    ├── results: List[RetrievalResult]
    ├── evidence_refs: List[MemoryEvidenceRef]
    └── support_level: str

Conflict Scoring
    ├── compute_conflict_penalty()
    ├── compute_support_strength()
    └── rerank_conflict_aware()
```

## Migration from Direct Engine Use

Before:

```python
results = engine.retrieve(query="Python", use_semantic=True)
for r in results:
    print(r.memory.content)
```

After:

```python
orchestrator = engine.create_orchestrator()
packet = orchestrator.retrieve_packet(query="Python", use_semantic=True)

for ref in packet.evidence_refs:
    print(f"{ref.memory_id}: confidence={ref.confidence}")

if packet.support_level in ("supported", "strong"):
    grounded = orchestrator.ground_answer("Python?", "A language", packet)
    if grounded["should_answer"]:
        print("Answer is grounded")
```
