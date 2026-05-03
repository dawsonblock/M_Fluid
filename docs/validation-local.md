# Local Validation Commands

> Commands to verify the memory system locally.

## Quick Validation

```bash
# Run all memory-related tests
PYTHONPATH=. uv run python -m pytest tests/ -q
```

## Phase-by-Phase Validation

### Phase 0: Baseline

```bash
# AST parse check (no syntax errors)
.venv/bin/python3 -c "import ast; ast.parse(open('fluid_memory/storage.py').read()); print('Syntax OK')"
.venv/bin/python3 -c "import ast; ast.parse(open('judge_memory/storage.py').read()); print('Syntax OK')"

# Import check (no dependency issues)
python3 -c "from judge_memory import JudgeMemoryService; print('✓ judge_memory imports')"
python3 -c "from fluid_memory import FluidMemoryEngine; print('✓ fluid_memory imports')"
```

### Phase 1: Judge Memory Search

```bash
# Test FTS sanitization and claim search with evidence metadata
PYTHONPATH=. uv run python -m pytest tests/test_judge_memory_search.py -v
```

**Expected:** 8 tests pass

### Phase 2: Mutation Safety

```bash
# Test mutation safety (protected fields, UUIDs, dedup)
PYTHONPATH=. uv run python -m pytest tests/test_fluid_mutation_safety.py -v
```

**Expected:** 15 tests pass

### Phase 3: Invalidation

```bash
# Test invalidation at retrieval boundary
PYTHONPATH=. uv run python -m pytest tests/test_fluid_invalidation.py -v
```

**Expected:** 9 tests pass

### Phase 4: Contradiction State

```bash
# Test contradiction changes multiple state fields
PYTHONPATH=. uv run python -m pytest tests/test_fluid_state.py tests/test_fluid_events.py -v
```

**Expected:** 22 tests pass

### Phase 12: Integration

```bash
# Test complete memory lifecycle
PYTHONPATH=. uv run python -m pytest tests/test_judge_fluid_memory_integration.py -v
```

**Expected:** 4 tests pass

## Complete Test Suite

```bash
# All tests
PYTHONPATH=. uv run python -m pytest tests/ -q
```

**Expected:** 148 tests pass

## Test Breakdown

| Test File | Count | Purpose |
|-----------|-------|---------|
| `test_fluid_imports.py` | 3 | Import verification |
| `test_fluid_decay.py` | 5 | Decay behavior |
| `test_fluid_storage.py` | 5 | Storage operations |
| `test_fluid_mutation.py` | 10 | Mutation operations |
| `test_fluid_mutation_safety.py` | 15 | Mutation safety (Phase 2) |
| `test_fluid_invalidation.py` | 9 | Invalidation (Phase 3) |
| `test_fluid_state.py` | 9 | State management |
| `test_fluid_events.py` | 13 | Event tracking |
| `test_fluid_retrieval.py` | 6 | Retrieval scoring |
| `test_fluid_scoring.py` | 5 | Scoring algorithms |
| `test_fluid_migration.py` | 5 | Schema migration |
| `test_judge_memory_migration.py` | 5 | FTS migration |
| `test_judge_memory_search.py` | 8 | Search improvements (Phase 1) |
| `test_judge_memory_isolated.py` | 18 | Core judge memory |
| `test_judge_fluid_memory_integration.py` | 4 | Integration (Phase 12) |
| **Total** | **148** | **All passing** |

## Manual Verification

### Evidence Vault Health

```python
import asyncio
import tempfile
from judge_memory import JudgeMemoryService, JudgeMemoryConfig

async def verify_vault():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = JudgeMemoryConfig(data_dir=tmpdir)
        # create_verified() checks vault health before returning
        service = await JudgeMemoryService.create_verified(config)
        print("✓ Vault health verified")
        await service.close()

asyncio.run(verify_vault())
```

### Mutation Safety

```python
from fluid_memory.models import MemoryItem
from fluid_memory.mutation import apply_state_delta

# Should raise ValueError for protected field
memory = MemoryItem(content="Test")
try:
    apply_state_delta(memory, {"memory_id": "new_id"}, strict_allowlist=True)
    print("✗ Failed to block protected field")
except ValueError as e:
    print(f"✓ Blocked: {e}")
```

### Invalidation Boundary

```python
import tempfile
from fluid_memory import FluidMemoryEngine, FluidMemoryConfig

with tempfile.TemporaryDirectory() as tmpdir:
    engine = FluidMemoryEngine(FluidMemoryConfig(db_path=f"{tmpdir}/test.db"))
    memory = engine.add_memory(content="Test content")
    
    # Invalidate
    engine.storage.invalidate(memory.memory_id, reason="Test")
    
    # Should not find in normal retrieval
    try:
        engine.get_memory(memory.memory_id)
        print("✗ Invalidated memory visible")
    except:
        print("✓ Invalidated memory hidden")
    
    # Should find with admin access
    retrieved = engine.storage.get_memory(memory.memory_id, include_invalidated=True)
    if retrieved and retrieved.invalidated_at:
        print("✓ Admin can see invalidated")
```

## Troubleshooting

### Import Errors

If `judge_memory` or `fluid_memory` fail to import:

```bash
# Check Python path
PYTHONPATH=. python3 -c "from judge_memory import JudgeMemoryService"
```

### Test Failures

If specific tests fail, run with verbose output:

```bash
PYTHONPATH=. uv run python -m pytest tests/test_specific.py -v --tb=short
```

### Database Lock Issues

If tests fail with "database is locked":

```bash
# Remove test databases
find . -name "*.db" -delete
```

## CI Equivalent

These commands are run in CI:

```bash
# Install dependencies
uv sync --dev --all-extras --reinstall

# Run tests
PYTHONPATH=. uv run python -m pytest tests/ -q

# Lint check
uv run ruff check .
```
