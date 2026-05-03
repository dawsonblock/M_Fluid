"""Deterministic benchmark helpers for memory retrieval validation."""

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from fluid_memory.config import FluidMemoryConfig
from fluid_memory.engine import FluidMemoryEngine


@dataclass
class BenchmarkCase:
    """A single benchmark test case."""

    name: str
    memories: List[Dict[str, Any]] = field(default_factory=list)
    query: str = ""
    expected_memory_ids: List[str] = field(default_factory=list)
    expected_tags: List[str] = field(default_factory=list)
    invalidated_ids: List[str] = field(default_factory=list)
    contradicted_ids: List[str] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    """Result of running a benchmark case."""

    case_name: str
    retrieved_ids: List[str] = field(default_factory=list)
    expected_hits: int = 0
    actual_hits: int = 0
    precision_at_k: float = 0.0
    invalidated_returned: int = 0
    contradicted_top_rank: bool = False
    passed: bool = False
    warnings: List[str] = field(default_factory=list)


def run_retrieval_benchmark(
    engine_or_config: Union[FluidMemoryEngine, FluidMemoryConfig],
    cases: List[BenchmarkCase],
    k: int = 5,
    isolated: bool = True,
) -> List[BenchmarkResult]:
    """Run deterministic retrieval benchmarks.

    Args:
        engine_or_config: FluidMemoryEngine to test, or config to create engines
        cases: List of BenchmarkCase definitions
        k: Top-k threshold for precision calculation
        isolated: If True, create fresh engine per case (no cross-contamination)

    Returns:
        List of BenchmarkResult objects
    """
    results = []

    for case in cases:
        if isolated:
            # Create fresh engine to prevent case cross-contamination
            fresh_engine = _create_fresh_engine(engine_or_config)
        else:
            # Use shared engine (legacy behavior, may cause contamination)
            fresh_engine = engine_or_config
        result = _run_single_case(fresh_engine, case, k)
        results.append(result)

    return results


def _create_fresh_engine(
    engine_or_config: Union[FluidMemoryEngine, FluidMemoryConfig]
) -> FluidMemoryEngine:
    """Create a fresh engine with temporary database.

    Args:
        engine_or_config: Either an engine (to extract config from) or a config

    Returns:
        Fresh FluidMemoryEngine with temporary database
    """
    if isinstance(engine_or_config, FluidMemoryEngine):
        # Extract config from existing engine
        config = engine_or_config.config
    else:
        config = engine_or_config

    # Create temporary directory for isolated test
    tmpdir = tempfile.mkdtemp()
    fresh_config = FluidMemoryConfig(
        data_dir=Path(tmpdir),
        db_path=Path(tmpdir) / "benchmark.db",
        retrieval_threshold=config.retrieval_threshold,
        max_results=config.max_results,
    )

    return FluidMemoryEngine(fresh_config, enable_audit=False)


def _resolve_id(id_or_case_id: str, memory_map: Dict[str, Any]) -> str:
    """Resolve a case_id or memory_id to actual memory_id."""
    if id_or_case_id in memory_map:
        mem = memory_map[id_or_case_id]
        return mem.memory_id
    return id_or_case_id


def _run_single_case(
    engine: FluidMemoryEngine,
    case: BenchmarkCase,
    k: int,
) -> BenchmarkResult:
    """Run a single benchmark case."""
    warnings: List[str] = []

    # Add memories for this case
    memory_map = {}
    for mem_data in case.memories:
        content = mem_data.get("content", "")
        tags = mem_data.get("tags", [])
        volatility = mem_data.get("volatility", 0.3)
        mem = engine.add_memory(
            content=content,
            tags=tags,
            volatility=volatility,
            detect_contradictions=False,
        )
        # Map both case_id and actual memory_id
        case_id = mem_data.get("case_id", mem.memory_id)
        memory_map[case_id] = mem
        memory_map[mem.memory_id] = mem

    # Resolve expected IDs from case_ids
    resolved_expected_ids = [
        _resolve_id(mid, memory_map) for mid in case.expected_memory_ids
    ]

    # Compute expected hits from tags if provided
    expected_ids = set(resolved_expected_ids)
    if case.expected_tags:
        for mem in memory_map.values():
            if any(tag in mem.tags for tag in case.expected_tags):
                expected_ids.add(mem.memory_id)

    expected_hits = len(expected_ids)

    # Apply invalidations (resolve case_ids first)
    resolved_invalidated_ids = [
        _resolve_id(mid, memory_map) for mid in case.invalidated_ids
    ]
    for mid in resolved_invalidated_ids:
        if mid in memory_map:
            engine.invalidate_memory(mid, reason="benchmark_invalidated")

    # Apply contradictions (resolve case_ids first)
    resolved_contradicted_ids = [
        _resolve_id(mid, memory_map) for mid in case.contradicted_ids
    ]
    for mid in resolved_contradicted_ids:
        if mid in memory_map:
            engine.contradict(mid, amount=0.5, reason="benchmark_contradicted")

    # Retrieve
    retrieved = engine.retrieve(query=case.query, limit=k)
    retrieved_ids = [r.memory.memory_id for r in retrieved]

    # Calculate metrics
    actual_hits = sum(1 for rid in retrieved_ids if rid in expected_ids)
    precision_at_k = actual_hits / len(retrieved_ids) if retrieved_ids else 0.0

    # Check for invalidated memories in results
    invalidated_returned = sum(
        1 for r in retrieved if r.memory.memory_id in resolved_invalidated_ids
    )
    if invalidated_returned > 0:
        warnings.append(
            f"{invalidated_returned} invalidated memories returned"
        )

    # Check if contradicted memory is top-ranked
    contradicted_top_rank = False
    if retrieved and retrieved[0].memory.memory_id in resolved_contradicted_ids:
        contradicted_top_rank = True
        warnings.append("Contradicted memory is top-ranked")

    # Determine pass/fail
    # Must have expected hits > 0 to pass, otherwise test is ill-defined
    passed = (
        expected_hits > 0
        and invalidated_returned == 0
        and actual_hits >= min(expected_hits, k)
        and not contradicted_top_rank
    )

    return BenchmarkResult(
        case_name=case.name,
        retrieved_ids=retrieved_ids,
        expected_hits=expected_hits,
        actual_hits=actual_hits,
        precision_at_k=precision_at_k,
        invalidated_returned=invalidated_returned,
        contradicted_top_rank=contradicted_top_rank,
        passed=passed,
        warnings=warnings,
    )
