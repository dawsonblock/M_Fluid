# Judge Memory Integration Summary

> **⚠️ DEPRECATED PATH:** The `m_flow.judge_memory` import path is deprecated.
> Use `from judge_memory import ...` (top-level package) instead.
> See `docs/MEMORY_SYSTEM_STATUS.md` for current architecture.

## What Was Delivered

A clean, bounded integration layer for the Judge app:

```
Judge app
  ↓
judge_memory (top-level package)
  ↓
├── Evidence Store (immutable files)
├── Claim Store (SQLite)
├── Search (keyword fallback)
└── Fluid Adapter (optional, disabled)
```

## Files Created

### Judge Memory Package
| File | Purpose |
|------|---------|
| `judge_memory/__init__.py` | Public API exports (canonical) |
| `m_flow/judge_memory/__init__.py` | Compatibility shim (deprecated) |
| `judge_memory/config.py` | JudgeMemoryConfig with safe defaults |
| `judge_memory/models.py` | EvidenceRecord, ClaimRecord, JudgeMemorySearchResult, TimelineEvent, SourcePacket, GroundedMemoryPacket |
| `judge_memory/service.py` | JudgeMemoryService - main API |
| `judge_memory/storage.py` | SQLite storage for evidence, claims, timeline |
| `judge_memory/evidence.py` | Immutable evidence file storage with SHA256 hashing |
| `judge_memory/claims.py` | ClaimsManager - claims linked to evidence |
| `judge_memory/source_registry.py` | SourceRegistry and DEFAULT_REGISTRY — canonical source trust profiles |
| `judge_memory/search.py` | Keyword search with fallback — uses SourceRegistry for authority scores |
| `judge_memory/fluid_adapter.py` | Thin adapter for `fluid_memory.FluidMemoryEngine` (standalone, optional) |
| `judge_memory/exceptions.py` | Custom exceptions |
| `judge_memory/README.md` | Integration guide |

### Tests Created
| File | Purpose |
|------|---------|
| `m_flow/tests/unit/judge_memory/test_judge_memory_storage.py` | SQLite storage tests |
| `m_flow/tests/unit/judge_memory/test_judge_memory_service.py` | Service API tests |
| `m_flow/tests/unit/judge_memory/test_judge_memory_fluid_disabled.py` | Fluid disabled by default tests |
| `m_flow/tests/unit/judge_memory/test_judge_memory_external_drive.py` | External drive path tests |

### V7 Bug Fixes (m_flow internals — unrelated to standalone judge_memory package)

> These fixes were applied to the m_flow stack at the same time. They do **not** affect
> the standalone `judge_memory` package, which no longer imports from `m_flow`.

| File | Fix |
|------|-----|
| `m_flow/memory/fluid/source_registry.py` | get_weights() now uses derived trust from structured fields |
| `m_flow/memory/fluid/source_registry.py` | DB loading uses `if x is not None else default` (not `or`) |
| `m_flow/memory/fluid/graph_access.py` | edge_type validation before Cypher interpolation |

## Public API (Canonical Path)

```python
from judge_memory import JudgeMemoryService, JudgeMemoryConfig

# Configure
config = JudgeMemoryConfig(
    data_dir="./judge_memory_data",
    enable_fluid_memory=False,  # Safe default
)

# Initialize
service = JudgeMemoryService(config)

# Ingest evidence (immutable, hash-deduplicated)
evidence = await service.ingest_evidence(
    raw_text="Court ruling text...",
    source_type="court_record",
    source_title="Smith v. Jones",
    jurisdiction="US-TX",
)

# Add claim (linked to evidence)
claim = await service.add_claim(
    evidence_id=evidence.evidence_id,
    claim_text="The court ruled for the plaintiff",
    claim_type="ruling",
    case_id="case_123",
)

# Search (keyword fallback, works without vector DB)
results = await service.search("court ruling plaintiff")

# Get timeline
timeline = await service.get_timeline(case_id="case_123")

# Get source packet (explainable trust profile)
packet = await service.get_source_packet(evidence.evidence_id)
# packet.authority, packet.verifiability, packet.legal_status_label

# Health check
health = await service.healthcheck()
```

## Safe Defaults

| Setting | Default | Why |
|---------|---------|-----|
| `enable_fluid_memory` | `False` | Disabled by default for safety |
| `enable_graph_retrieval` | `False` | No external DB required |
| `enable_vector_retrieval` | `False` | No vector DB required |
| `enable_llm_contradiction` | `False` | LLM detection disabled |
| `require_review_for_legal_claims` | `True` | Extra safety |
| `allow_raw_cypher` | `False` | Security |
| `allow_mutation_tools` | `False` | Security |

## Immutable Evidence Principle

1. SHA256 hash computed from raw_text
2. Duplicate hash returns existing record (no overwrite)
3. Files stored as JSON with metadata
4. Never overwrite existing evidence files

## How to Integrate into Judge ZIP

1. Copy `judge_memory/` directory to your Judge app (top-level package)
2. Import: `from judge_memory import JudgeMemoryService, JudgeMemoryConfig`
3. Configure paths (can be external drive)
4. Use the API as shown above
5. Fluid memory can be enabled later without code changes (requires m_flow installed)

## Known Limitations

1. Search is keyword-based only (no semantic/vector yet)
2. Graph retrieval requires external Neo4j/Kuzu
3. Fluid memory is minimal integration (disabled by default)
4. LLM contradiction detection is off by default
5. Unit tests only (no integration tests due to Python 3.9 compatibility issues in full m_flow stack)

## Tests

Tests are written but cannot run due to Python 3.9 compatibility issues in the broader m_flow codebase (union syntax `|`). The judge_memory module itself is Python 3.9+ compatible.

To run tests in a compatible environment:
```bash
python -m pytest m_flow/tests/unit/judge_memory -v
```

## Future Upgrades (After Integration)

Enable features by changing config:
- `enable_fluid_memory=True` - Enable fluid scoring
- `enable_vector_retrieval=True` - Add semantic search
- `enable_graph_retrieval=True` - Add graph traversal
- `enable_llm_contradiction=True` - Add contradiction detection

## Status

**Integration-ready local memory subsystem.**
Not production-grade. Ready for Judge app integration.
