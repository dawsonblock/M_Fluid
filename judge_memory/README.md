# Judge Memory

Standalone legal/evidence memory subsystem for Judge applications.

## Overview

Judge Memory is a bounded, importable memory module designed for legal and evidence-tracking applications. It operates independently without requiring the full M-flow framework.

## Features

- **SQLite local storage** - No external database required
- **Immutable evidence preservation** - SHA256 hash deduplication
- **Claims linked to evidence** - Mutable interpretations of immutable records
- **Source trust profiles** - Explainable authority scoring
- **Timeline view** - Chronological case/person/judge history
- **Optional fluid memory** - M-flow integration when available
- **Safe defaults** - Fluid memory disabled by default

## Installation

Requires only:
- Python 3.8+ (tested on 3.8, 3.9, 3.10, 3.11, 3.12)
- Pydantic

No structlog, no graph database, no vector database required.

```bash
pip install pydantic
```

### Python 3.11+ Ready

The `judge_memory` package uses traditional `Optional[Type]` syntax for maximum
compatibility across Python versions. This ensures it works on Python 3.8+
including 3.11 and 3.12 without requiring any code changes.

## Quick Start

```python
from judge_memory import JudgeMemoryService, JudgeMemoryConfig

# Create service
config = JudgeMemoryConfig(
    data_dir="./judge_memory_data",
    enable_fluid_memory=False,  # Safe default
)
service = JudgeMemoryService(config)

# Ingest evidence
evidence = await service.ingest_evidence(
    raw_text="Court ruling text...",
    source_type="court_record",
    source_title="Smith v. Jones",
    jurisdiction="US-TX",
)

# Add claim
claim = await service.add_claim(
    evidence_id=evidence.evidence_id,
    claim_text="The court ruled in favor of the plaintiff",
    claim_type="ruling",
    case_id="case_123",
)

# Search
results = await service.search("court ruling plaintiff")

# Get source packet (trust profile)
packet = await service.get_source_packet(evidence.evidence_id)
print(f"Authority: {packet.authority}")
print(f"Legal status: {packet.legal_status_label}")
```

## Architecture

### Evidence vs Claims

**Evidence** (immutable):
- SHA256 content hash for deduplication
- Source metadata (type, jurisdiction, URL)
- File storage with hash-based naming
- Never modified after ingestion

**Claims** (mutable):
- Linked to parent evidence
- Confidence scoring
- Review workflow (active, under_review, retracted, confirmed)
- Contradiction tracking

### Source Trust Profiles

Hardcoded profiles for common source types:

| Source Type | Authority | Legal Status |
|-------------|-----------|--------------|
| court_record | 0.90 | primary_authority |
| government_data | 0.85 | official_record |
| police_release | 0.70 | official_statement |
| academic_paper | 0.75 | expert_opinion |
| expert_report | 0.80 | expert_testimony |
| witness_statement | 0.60 | eyewitness_account |
| mainstream_news | 0.60 | press_report |
| blog_social | 0.30 | unverified_source |

### Optional Fluid Memory Integration

When M-flow is installed and `enable_fluid_memory=True`:
- Calls `m_flow.memory.fluid` engine for state tracking
- Lane-specific decay (legal, trust, interest, attention)
- Contradiction pressure tracking
- Activation propagation

When M-flow is unavailable:
- Uses hardcoded source profiles (no external dependencies)
- Fluid state tracking disabled
- Core evidence/claims functionality preserved
- No error - graceful degradation

## Database Schema

### evidence_records
- evidence_id (PK)
- content_hash (unique)
- source_type, source_url, source_title
- jurisdiction, published_at
- file_path, metadata, ingested_at

### claim_records
- claim_id (PK)
- evidence_id (FK)
- claim_text, claim_type
- case_id, judge_id, person_id, entity_id
- confidence, status, tags, metadata
- created_at, updated_at

### timeline_events
- event_id (PK)
- event_type, event_date, description
- evidence_id, claim_id, case_id
- judge_id, person_id, entity_id
- jurisdiction, metadata

## Configuration

```python
JudgeMemoryConfig(
    data_dir="./judge_memory_data",      # Storage directory
    sqlite_path=None,                     # Explicit DB path (optional)
    enable_fluid_memory=False,            # Safe default
    min_claim_confidence=0.0,             # Confidence threshold
    max_claims_per_evidence=100,          # Limit per evidence
    enable_audit=True,                    # Audit logging
)
```

## API Reference

### JudgeMemoryService

**Evidence Operations:**
- `ingest_evidence(raw_text, source_type, ...)` → EvidenceRecord
- `get_source_packet(evidence_id)` → SourcePacket
- `get_evidence_content(evidence_id)` → str

**Claim Operations:**
- `add_claim(evidence_id, claim_text, ...)` → ClaimRecord
- `review_claim(claim_id, status, ...)` → ClaimRecord

**Search Operations:**
- `search(query, entity_id, case_id, judge_id, ...)` → List[JudgeMemorySearchResult]
- `get_timeline(entity_id, case_id, judge_id, ...)` → List[JudgeMemorySearchResult]

**Lifecycle:**
- `close()` - Release resources

## Testing

```python
# Test without M-flow installed
python -c "from judge_memory import JudgeMemoryService; print('OK')"

# Test with async
import asyncio
async def test():
    service = JudgeMemoryService(JudgeMemoryConfig())
    # ... test operations
    await service.close()

asyncio.run(test())
```

## Safety Features

1. **Duplicate prevention** - Content hash deduplication
2. **Immutable evidence** - Never overwrites source files
3. **Claim review workflow** - Explicit status transitions
4. **Fail-safe fluid memory** - Disabled by default, graceful degradation
5. **No silent failures** - Structured logging with NullHandler

## Migration from m_flow.judge_memory

The old import path:
```python
from m_flow.judge_memory import JudgeMemoryService
```

New import path:
```python
from judge_memory import JudgeMemoryService
```

The API remains compatible. The old `m_flow/judge_memory/` directory can be removed or kept as a compatibility shim.

## License

Same as M-flow project.
