# Judge Memory — Architecture Target

> Freeze document. Describes the intended design of the `judge_memory` +
> `fluid_memory` subsystem. Deviations from these rules are defects.

---

## Four Canonical Rules

### Rule 1 — No m_flow Runtime Dependency

`judge_memory` and `fluid_memory` **must never import from `m_flow`**.

```python
# FORBIDDEN in any judge_memory or fluid_memory file:
from m_flow import ...
import m_flow
```

The only permitted runtime dependency on the rest of this repo is:

```python
from fluid_memory import FluidMemoryEngine, FluidMemoryConfig
```

This import is itself **optional** — the entire `judge_memory` package must
function correctly (SQLite storage, full-text search, claim management) when
`fluid_memory` is absent or disabled via `enable_fluid_memory=False`.

**Verification gate** (must pass before any release):

```bash
python -c "
import sys
from judge_memory import JudgeMemoryService
assert 'm_flow' not in sys.modules, f'm_flow leaked: {[k for k in sys.modules if k.startswith(\"m_flow\")]}'
print('OK: m_flow not in sys.modules')
"
```

---

### Rule 2 — Evidence Is Immutable

Once ingested, an `EvidenceRecord` **must never be mutated or deleted** by
application code. The content hash (`SHA-256` of raw text) is the canonical
deduplication key.

Permitted operations on evidence:
- `ingest_evidence()` — creates or returns existing record (idempotent)
- `get_evidence()` — read-only retrieval
- `search()` — full-text search (does not modify records)
- `touch_evidence()` (in `FluidMemoryAdapter`) — updates *fluid state only*,
  never touches the `EvidenceRecord` itself

Forbidden:
- `UPDATE` on the `evidence` table outside of Alembic migrations
- `DELETE` on the `evidence` table (soft-delete via status is acceptable)
- Overwriting `content_hash` or `ingested_at`

---

### Rule 3 — Source Authority Has One Source of Truth

Numerical trust weights and source type labels live in exactly **one place**:
`judge_memory/source_registry.py → SourceRegistry`.

`judge_memory/search.py` must not define its own `SOURCE_AUTHORITY` dict.
`judge_memory/fluid_adapter.py` must not define its own weights dict.
Both must call `SourceRegistry.get_profile(source_type)`.

Until `SourceRegistry` is wired in, the `HARDCODED_SOURCE_PROFILES` dict in
`fluid_adapter.py` is the temporary single copy — `search.py` must delegate to
it via the injected `source_registry` parameter rather than maintaining a
separate dict.

---

### Rule 4 — Contradiction Pressure Never Hides Evidence

The contradiction detection and decay system in `fluid_memory` is allowed to
lower *salience* and *confidence* scores on `MemoryItem`s. It is **not
allowed** to:

- Delete a `MemoryItem` whose `source_refs` contain a judge evidence ID
- Set `is_invalidated = True` on a legal or court record item without a
  corroborating human review action
- Cascade a contradiction event into changes on the `EvidenceRecord` or
  `ClaimRecord` tables

---

## Package Dependency Graph

```
judges_app (consumer)
    └── judge_memory
            ├── fluid_memory          (optional, same repo, no m_flow)
            ├── SQLite / FTS5         (stdlib sqlite3)
            └── pydantic, fastapi     (third-party)
```

`fluid_memory` has no dependencies on `judge_memory` (one-way only).

---

## Source Type Registry

| Source Type      | Authority | Legal Lane | Default Claim Status |
|------------------|-----------|------------|----------------------|
| court_record     | 0.90      | legal      | presumed_valid       |
| government_data  | 0.85      | legal      | presumed_valid       |
| police_release   | 0.70      | trust      | needs_verification   |
| expert_report    | 0.80      | interest   | needs_verification   |
| academic_paper   | 0.75      | interest   | needs_verification   |
| witness_statement| 0.60      | interest   | needs_verification   |
| mainstream_news  | 0.60      | interest   | needs_verification   |
| blog_social      | 0.30      | attention  | unverified           |
| unknown          | 0.50      | attention  | needs_verification   |

Decay lane determines how quickly a memory item loses salience over time:
`legal` (slowest) → `trust` → `interest` → `attention` (fastest).

---

## Verification Checklist

Before any PR that touches `judge_memory/` or `fluid_memory/`:

- [ ] `python -c "from judge_memory import JudgeMemoryService; import sys; assert 'm_flow' not in sys.modules"` passes
- [ ] `PYTHONPATH=. pytest m_flow/tests/unit/test_judge_memory*.py -v` passes
- [ ] `from judge_memory import SourceRegistry` works
- [ ] `from judge_memory import GroundedMemoryPacket` works
- [ ] All source authority weights in `SourceRegistry` match the table above
