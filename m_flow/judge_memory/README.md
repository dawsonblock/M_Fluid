# Judge Memory - Integration-Ready Local Memory Subsystem

A bounded, importable memory module for the Judge app with:
- **SQLite local storage** (no external DB required)
- **Immutable evidence preservation** (SHA256 hash-deduplicated)
- **Claims linked to evidence** (no orphan claims)
- **Basic search fallback** (works without vector/graph DB)
- **Optional fluid memory scoring** (disabled by default)
- **Explainable source profiles** (structured trust fields)

> **Status**: Integration-ready local subsystem. Not production-grade.

## Quick Start (Isolated Mode - No Full M-flow Required)

```python
# Use top-level judge_memory for isolated mode (no structlog/dependencies)
from judge_memory import JudgeMemoryService, JudgeMemoryConfig

# Configure (safe defaults)
config = JudgeMemoryConfig(
    data_dir="./judge_memory_data",
    enable_fluid_memory=False,  # Safe default
)

# Initialize service
memory = JudgeMemoryService(config)

# Ingest evidence (immutable, hash-deduplicated)
evidence = await memory.ingest_evidence(
    raw_text="Court ruling text...",
    source_type="court_record",
    source_title="Smith v. Jones",
    jurisdiction="US-TX",
)

# Add claim linked to evidence
claim = await memory.add_claim(
    evidence_id=evidence.evidence_id,
    claim_text="The court ruled in favor of the plaintiff",
    claim_type="ruling",
    case_id="case_123",
)

# Search (keyword fallback, works without vector DB)
results = await memory.search("court ruling plaintiff")

# Get source packet with trust profile
packet = await memory.get_source_packet(evidence.evidence_id)
print(f"Authority: {packet.authority}")
print(f"Legal status: {packet.legal_status_label}")
print(f"Trust score: {packet.authority * 0.3 + packet.verifiability * 0.3 + packet.originality * 0.2 + packet.independence * 0.2}")
```

## External Drive Support

```python
config = JudgeMemoryConfig(
    data_dir=Path("/Volumes/JudgeMemory"),
    evidence_dir=Path("/Volumes/JudgeMemory/evidence"),
    sqlite_path=Path("/Volumes/JudgeMemory/judge_memory.sqlite"),
    enable_fluid_memory=False,
)
```

## Safe Defaults

| Setting | Default | Description |
|---------|---------|-------------|
| `enable_fluid_memory` | `False` | Fluid scoring disabled by default |
| `enable_graph_retrieval` | `False` | Graph DB not required |
| `enable_vector_retrieval` | `False` | Vector DB not required |
| `enable_llm_contradiction` | `False` | LLM contradiction detection off |
| `require_review_for_legal_claims` | `True` | Extra safety for legal claims |
| `allow_raw_cypher` | `False` | Raw Cypher disabled |
| `allow_mutation_tools` | `False` | Mutation tools disabled |

## Import Path Options

### Isolated Mode (Recommended for Judge ZIP)

Use when you want judge_memory without full M-flow dependencies:

```python
from judge_memory import JudgeMemoryService, JudgeMemoryConfig
```

**Requirements:** Only requires sqlite3, pydantic (no structlog, no full m_flow stack)

### Legacy Shim (Requires Full M-flow)

Use when you're already running inside full M-flow environment:

```python
from m_flow.judge_memory import JudgeMemoryService, JudgeMemoryConfig
```

**Requirements:** Full M-flow stack including structlog, dotenv, and all API modules.
**Note:** The shim emits a DeprecationWarning on import.

## Source Trust Profiles

Source types have structured trust profiles:

```python
from m_flow.memory.fluid.source_registry import _HARDCODED_FALLBACK

profile = _HARDCODED_FALLBACK["court_record"]
print(profile.authority)          # 1.00
print(profile.verifiability)      # 0.95
print(profile.legal_status_label)  # "official_record"

# Derived trust (weighted average)
trust = profile.derive_trust()  # 0.30*authority + 0.30*verifiability + 0.20*originality + 0.20*independence
```

## Testing

```bash
python -m pytest m_flow/tests/unit/judge_memory -v
```

## Architecture

```
Judge app
  ↓
JudgeMemoryService
  ↓
├── EvidenceStorage (immutable files)
├── JudgeMemoryStorage (SQLite)
├── ClaimsManager (claim → evidence links)
├── JudgeMemorySearch (keyword fallback)
└── FluidMemoryAdapter (optional, disabled by default)
```

## Files

| File | Purpose |
|------|---------|
| `__init__.py` | Public API exports |
| `config.py` | JudgeMemoryConfig |
| `models.py` | EvidenceRecord, ClaimRecord, etc. |
| `service.py` | JudgeMemoryService main API |
| `storage.py` | SQLite storage layer |
| `evidence.py` | Immutable file storage with hashing |
| `claims.py` | Claim management |
| `search.py` | Search implementation |
| `fluid_adapter.py` | Optional fluid memory adapter |
| `exceptions.py` | Custom exceptions |

## Known Limitations

1. **Search**: Keyword-based fallback only (no vector/semantic search yet)
2. **Graph**: No graph traversal without external Neo4j/Kuzu
3. **Fluid**: Disabled by default, minimal integration
4. **LLM**: No contradiction detection by default
5. **Tests**: Unit tests only, no integration tests

## Integration into Judge ZIP

1. Copy `m_flow/judge_memory/` to your Judge app
2. Import and use the API
3. Evidence stored immutably on external drive if configured
4. Fluid memory can be enabled later without code changes

## Future Upgrades (Post-Integration)

- Enable vector search: `enable_vector_retrieval=True`
- Enable graph retrieval: `enable_graph_retrieval=True`
- Enable fluid scoring: `enable_fluid_memory=True`
- Add LLM contradiction: `enable_llm_contradiction=True`
