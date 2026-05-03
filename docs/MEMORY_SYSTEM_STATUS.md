# Memory System Status

> Internal reference for AI assistants and contributors. Last updated: 2024

## System Overview

This repository contains multiple memory subsystems with different purposes and maturity levels. This document establishes the canonical architecture and current truth.

## Memory Subsystems

### 1. `judge_memory/` — Canonical Judge App Integration

**Purpose:** Standalone legal/evidence memory for Judge applications.

**Status:** Testable local prototype. Not production-grade.

**Location:** `/judge_memory/` (top-level package)

**Dependencies:** Python stdlib + Pydantic only. No structlog, no graph DB, no vector DB required.

**Canonical Import:**
```python
from judge_memory import JudgeMemoryService, JudgeMemoryConfig
```

**Core Features (Implemented):**
- ✅ SQLite storage for evidence, claims, timeline
- ✅ SHA256 content hashing for evidence deduplication
- ✅ Immutable evidence records
- ✅ Mutable claims linked to evidence
- ✅ Source trust profiles (hardcoded + override support)
- ✅ FTS5 search with automatic triggers for claims and evidence
- ✅ Claim search returns evidence metadata (source_type, jurisdiction)
- ✅ FTS query sanitization (prevents crashes from special chars)
- ✅ Timeline queries
- ✅ Orphaned claim prevention (validates evidence_id exists)
- ✅ Evidence vault health check (create_verified factory)
- ✅ Keyword-only search (FTS5); no vector/embedding search
- ✅ Heuristic contradiction flagging (tag + confidence similarity)
- ✅ Structured JSON audit logging

**Features (Implemented):**
- ✅ Fluid memory integration — uses standalone `fluid_memory.FluidMemoryEngine`; hardcoded profiles are a local fallback only. Zero dependency on `m_flow`.

**Features (Not Implemented):**
- ❌ Distributed storage. Single-node SQLite only.

---

### 2. `m_flow/memory/fluid/` — Advanced Fluid State Engine

**Purpose:** Dynamic memory state engine with activation, decay, confidence, salience.

**Status:** Architecturally promising, integrated with M-flow.

**Location:** `/m_flow/memory/fluid/`

**Dependencies:** Full M-flow stack (structlog, optional graph DB, etc.)

**Usage:**
```python
from m_flow.memory.fluid import FluidMemoryEngine
```

**Core Features:**
- ✅ Lane-specific decay (legal, trust, interest, attention)
- ✅ Contradiction pressure tracking
- ✅ Activation propagation
- ✅ Source lineage and legal weight
- ✅ Retrieval scoring

**Integration:** Called by `judge_memory/fluid_adapter.py` when `enable_fluid_memory=True` and m_flow available.

---

### 3. `fluid_memory/` — Standalone Prototype

**Purpose:** Standalone adaptive memory (SQLite/Pydantic) without M-flow dependencies.

**Status:** Restored as standalone package. Has useful decay/retrieval concepts.

**Location:** `/fluid_memory/` (top-level package)

**Dependencies:** Python stdlib + Pydantic only.

**Canonical Import:**
```python
from fluid_memory import FluidMemoryEngine, FluidMemoryConfig
```

**Core Features:**
- ✅ Memory storage with salience/confidence/volatility/stability
- ✅ Schema migration with ALTER TABLE for new columns
- ✅ Checksum and invalidation support
- ✅ Invalidation at retrieval boundary (hidden from normal search)
- ✅ Decay lanes (time, trust, interest, attention)
- ✅ Retrieval with fluid scoring
- ✅ Reinforcement and contradiction tracking
- ✅ Contradiction changes multiple state fields (confidence, volatility, stability, attention_salience)
- ✅ Mutation safety (protected fields, allowlist, UUID event IDs)
- ✅ Duplicate content detection
- ✅ Admin retrieval methods for invalidated memories
- ✅ Local hashed character-bigram similarity search
- ✅ Heuristic contradiction flagging (tag + confidence similarity)
- ✅ Structured JSON audit logging

**Known Issues (Fixed):**
- ✅ Uses UUID4 (not Python hash()) for event IDs
- ✅ SHA256 hashing for content
- ✅ Mutation safety enforced
- ✅ Invalidation filtering implemented
- ✅ Semantic search implemented
- ✅ Audit logging implemented

---

### 4. `m_flow/judge_memory/` — Deprecated Compatibility

**Purpose:** Backward compatibility shim.

**Status:** Deprecated. Do not use for new code.

**Location:** `/m_flow/judge_memory/`

**Import (Deprecated):**
```python
from m_flow.judge_memory import JudgeMemoryService  # DeprecationWarning
```

**Behavior:** Re-exports from top-level `judge_memory/`. Triggers m_flow import chain.

---

## Architectural Rules

1. **Evidence is immutable.** Never modify after ingestion. SHA256 hash ensures deduplication.
2. **Claims are mutable.** Linked to evidence, can be reviewed, retracted, confirmed.
3. **Fluid state is mutable.** Tracks activation, confidence, salience over time.
4. **Truth status is controlled.** Claims have explicit status workflow.
5. **Fluid engine never rewrites evidence.** New evidence touches state, changes confidence/pressure/activation, but immutable evidence records remain preserved.

## Import Boundaries

| Import | Dependencies | Use Case |
|--------|--------------|----------|
| `from judge_memory import ...` | stdlib + Pydantic | Judge app integration (recommended) |
| `from fluid_memory import ...` | stdlib + Pydantic | Standalone adaptive memory |
| `from m_flow.memory.fluid import ...` | Full M-flow stack | Advanced fluid with M-flow |
| `from m_flow.judge_memory import ...` | Full M-flow stack | Deprecated, avoid |

## Validation Commands

```bash
# Test judge_memory clean import (no structlog)
python3 -c "import sys; from judge_memory import JudgeMemoryService; assert 'structlog' not in sys.modules; print('✓ judge_memory imports cleanly')"

# Test fluid_memory import
python3 -c "from fluid_memory import FluidMemoryEngine; print('✓ fluid_memory imports cleanly')"

# Test orphaned claim prevention
python3 -c "
import asyncio
from judge_memory import JudgeMemoryService, JudgeMemoryConfig
from judge_memory.exceptions import EvidenceNotFoundError

async def test():
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tmpdir:
        config = JudgeMemoryConfig(data_dir=tmpdir)
        service = JudgeMemoryService(config)
        try:
            await service.add_claim('fake_id', 'test')
            print('✗ Orphan check failed')
        except EvidenceNotFoundError:
            print('✓ Orphan prevention works')
        finally:
            await service.close()

asyncio.run(test())
"

# Run full test suite
python3 test_judge_memory_isolated.py
```

## Current Limitations (Truthful)

- **Not production-grade.** Local prototype suitable for integration testing.
- **Search is keyword-based.** No semantic/vector retrieval yet.
- **Fluid integration partial.** Hardcoded profiles when m_flow unavailable.
- **Audit logging basic.** Not enterprise audit-grade.
- **No distributed storage.** Single-node SQLite only.

## Documentation Status

| File | Status | Notes |
|------|--------|-------|
| `docs/MEMORY_SYSTEM_STATUS.md` | ✅ Current | This file - single source of truth |
| `JUDGE_MEMORY_ISOLATION_SUMMARY.txt` | ✅ Current | Removed "production ready" claims |
| `JUDGE_MEMORY_INTEGRATION.md` | ✅ Current | Updated to `judge_memory` import path |
| `judge_memory/README.md` | ✅ Current | Documented fluid as "calls when available" |
| `docs/FLUID_MEMORY_CORE.md` | ✅ Current | Updated status and architecture notes |

---

*This document is the canonical reference. When in doubt, believe this file over other docs that may be stale.*
