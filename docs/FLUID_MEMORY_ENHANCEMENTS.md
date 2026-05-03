# Fluid Memory Enhancements (Version 16)

Complete feature documentation for Version 16 comprehensive enhancements.

## 1. Operational Helpers

### Health Checks

Monitor system health programmatically:

```python
from fluid_memory import FluidMemoryEngine, FluidMemoryConfig

engine = FluidMemoryEngine(FluidMemoryConfig(db_path="memory.db"))

# Check overall health
if engine.health.is_healthy():
    print("System healthy")

# Get detailed checks
checks = engine.health.check_all()
for component, status in checks.items():
    print(f"{component}: {status.healthy} - {status.message}")

# Check specific components
storage_status = engine.health.check_storage()
checksum_status = engine.health.check_checksums(sample_size=10)
```

### Metrics Collection

Track operation performance:

```python
# Start timing an operation
engine.metrics.start_operation("search_123")

# Perform operation
results = engine.retrieve(query="AI")

# End and record
engine.metrics.end_operation(
    operation_id="search_123",
    operation_type="retrieve",
    success=True,
    details={"result_count": len(results)}
)

# Get summary report
summary = engine.metrics.get_summary()
print(f"Total operations: {summary['total_operations']}")
print(f"Avg duration: {summary['avg_duration_ms']:.2f}ms")
print(f"Success rate: {summary['overall_success_rate']:.1%}")
```

### Batch Operations

Efficient bulk operations:

```python
# Add multiple memories
contents = ["Memory 1", "Memory 2", "Memory 3"]
result = engine.batch.add_memories(contents, detect_contradictions=False)
print(f"Added {result.success_count} memories")

# Invalidate multiple memories
memory_ids = ["mem_abc", "mem_def"]
result = engine.batch.invalidate_memories(memory_ids, reason="Expired")

# Verify checksums in batch
result = engine.batch.verify_memories(memory_ids)
for detail in result.details:
    print(f"{detail['memory_id']}: {'valid' if detail['valid'] else 'INVALID'}")

# Link memories
source_id = "mem_source"
target_ids = ["mem_a", "mem_b", "mem_c"]
result = engine.batch.link_memories(source_id, target_ids, link_type="related")
```

## 2. Retrieval Quality

### Temporal Boost

Recent memories are automatically weighted higher:

- **< 1 day old**: 1.2x boost
- **< 1 week old**: 1.1x boost  
- **1 week - 1 month**: 1.0x (neutral)
- **> 1 month**: 0.9x (slight penalty)

```python
# Enabled by default
results = engine.retrieve(query="AI", limit=10)

# Disable if needed
results = engine.retrieve(
    query="AI",
    enable_temporal_boost=False
)
```

### Deduplication

Near-duplicate results are automatically filtered:

```python
# Enabled by default (85% similarity threshold)
results = engine.retrieve(query="software", limit=10)
# Returns unique results only

# Disable for all raw results
results = engine.retrieve(
    query="software",
    enable_deduplication=False
)
```

### Maximal Marginal Relevance (MMR)

Balance relevance with result diversity:

```python
# Enable MMR with balanced setting
results = engine.retrieve(
    query="AI",
    limit=5,
    enable_mmr=True,
    mmr_lambda=0.5  # 0 = diversity only, 1 = relevance only
)

# Favor diversity (good for exploration)
results = engine.retrieve(
    query="machine learning",
    enable_mmr=True,
    mmr_lambda=0.3
)

# Favor relevance (good for precision)
results = engine.retrieve(
    query="specific topic",
    enable_mmr=True,
    mmr_lambda=0.8
)
```

### Combined Options

Use all enhancements together:

```python
results = engine.retrieve(
    query="artificial intelligence",
    limit=10,
    enable_temporal_boost=True,
    enable_deduplication=True,
    enable_mmr=True,
    mmr_lambda=0.5
)
```

## 3. Orchestration

### Pagination

Process large memory sets in batches:

```python
# apply_decay now paginates automatically
# Processes all memories regardless of count
engine.apply_decay(days=1.0, batch_size=1000)

# Storage supports offset for custom pagination
page1 = engine.storage.get_all(limit=100, offset=0)
page2 = engine.storage.get_all(limit=100, offset=100)
```

## 4. API Reference

### HealthStatus

```python
@dataclass
class HealthStatus:
    healthy: bool
    component: str
    message: str
    details: Dict[str, Any]
    timestamp: float
```

### BatchResult

```python
@dataclass
class BatchResult:
    success_count: int
    failure_count: int
    errors: List[Tuple[str, str]]  # (id, error_message)
    details: List[Any]
```

### OperationMetric

```python
@dataclass
class OperationMetric:
    operation: str
    duration_ms: float
    success: bool
    timestamp: float
    details: Dict[str, Any]
```

## 5. Configuration

Default retrieval settings can be configured:

```python
config = FluidMemoryConfig(
    db_path="memory.db",
    max_results=20,
    retrieval_threshold=0.3
)
engine = FluidMemoryEngine(config)
```

## Migration Guide

Version 16 is fully backward compatible. All new features are opt-in or have sensible defaults:

- `retrieve()` works exactly as before if no new parameters provided
- Temporal boost and deduplication are enabled by default
- MMR is disabled by default
- Health, metrics, and batch operations are available but not required
