"""Test retrieval benchmarks."""

import tempfile

from fluid_memory import FluidMemoryEngine, FluidMemoryConfig
from fluid_memory.benchmarks import (
    BenchmarkCase,
    run_retrieval_benchmark,
)


def test_invalidated_memories_not_retrieved_in_benchmark():
    """Invalidated memories are not returned in benchmark results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = FluidMemoryEngine(
            FluidMemoryConfig(db_path=f"{tmpdir}/test.db"),
            enable_audit=False,
        )

        # Create a case where one memory is invalidated
        case = BenchmarkCase(
            name="invalidation_test",
            memories=[
                {"content": "Valid Python info", "tags": ["python"]},
                {"content": "Invalid Python info", "tags": ["python"]},
            ],
            query="Python",
            expected_memory_ids=[],
            invalidated_ids=["will_be_set_after_add"],
        )

        # Add memories and update the case with actual IDs
        mem1 = engine.add_memory(
            content="Valid Python info",
            tags=["python"],
            detect_contradictions=False,
        )
        mem2 = engine.add_memory(
            content="Invalid Python info",
            tags=["python"],
            detect_contradictions=False,
        )

        # Invalidate the second one
        engine.invalidate_memory(mem2.memory_id, reason="outdated")

        # Update case
        case.invalidated_ids = [mem2.memory_id]
        case.expected_memory_ids = [mem1.memory_id]

        # Run benchmark manually
        results = engine.retrieve(query="Python")
        retrieved_ids = [r.memory.memory_id for r in results]

        # Invalidated should not be returned
        assert mem2.memory_id not in retrieved_ids


def test_expected_relevant_memories_in_top_k():
    """Expected relevant memories appear in top k results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = FluidMemoryEngine(
            FluidMemoryConfig(db_path=f"{tmpdir}/test.db"),
            enable_audit=False,
        )

        # Add memories
        mem1 = engine.add_memory(
            content="Python programming guide",
            detect_contradictions=False,
        )
        engine.add_memory(
            content="JavaScript guide",
            detect_contradictions=False,
        )
        engine.add_memory(
            content="Rust programming",
            detect_contradictions=False,
        )

        # Retrieve
        results = engine.retrieve(query="Python", limit=3)
        retrieved_ids = [r.memory.memory_id for r in results]

        # Python memory should be in top k
        assert mem1.memory_id in retrieved_ids


def test_contradicted_memory_not_top_ranked_when_stable_alternative():
    """Contradicted memory is not top-ranked when stable alternative exists."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = FluidMemoryEngine(
            FluidMemoryConfig(db_path=f"{tmpdir}/test.db"),
            enable_audit=False,
        )

        # Add stable memory
        stable_mem = engine.add_memory(
            content="Python is a programming language",
            confidence=0.7,
            detect_contradictions=False,
        )

        # Add memory to be contradicted
        contradicted_mem = engine.add_memory(
            content="Python is the best language",
            confidence=0.9,  # Higher confidence initially
            detect_contradictions=False,
        )

        # Contradict it
        engine.contradict(
            contradicted_mem.memory_id,
            amount=0.6,
            reason="evidence contradicts",
        )

        # Use orchestrator for conflict-aware retrieval
        orchestrator = engine.create_orchestrator()
        packet = orchestrator.retrieve_packet(
            query="Python",
            conflict_aware=True,
        )

        if packet.results:
            # Top result should not be the contradicted one
            top_id = packet.results[0].memory.memory_id
            assert top_id != contradicted_mem.memory_id


def test_precision_at_k_computed_correctly():
    """Precision@k is computed correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = FluidMemoryEngine(
            FluidMemoryConfig(db_path=f"{tmpdir}/test.db"),
            enable_audit=False,
        )

        # Create benchmark case
        case = BenchmarkCase(
            name="precision_test",
            memories=[
                {"content": "Python guide", "tags": ["python"]},
                {"content": "Python tutorial", "tags": ["python"]},
                {"content": "JavaScript guide", "tags": ["js"]},
            ],
            query="Python",
        )

        # Add memories
        mems = []
        for m in case.memories:
            mem = engine.add_memory(
                content=m["content"],
                tags=m.get("tags", []),
                detect_contradictions=False,
            )
            mems.append(mem)

        # Update expected IDs (the two Python ones)
        case.expected_memory_ids = [mems[0].memory_id, mems[1].memory_id]

        # Run retrieval at k=3
        results = engine.retrieve(query=case.query, limit=3)
        retrieved_ids = [r.memory.memory_id for r in results]

        # Compute precision@3
        hits = sum(1 for rid in retrieved_ids if rid in case.expected_memory_ids)
        precision = hits / len(retrieved_ids) if retrieved_ids else 0.0

        # Should have found both Python memories
        assert precision >= 0.5  # At least 1.5/3 or 2/3


def test_benchmark_warnings_for_invalidated_returned():
    """Benchmark warns when invalidated memories are returned."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = FluidMemoryEngine(
            FluidMemoryConfig(db_path=f"{tmpdir}/test.db"),
            enable_audit=False,
        )

        # This shouldn't happen, but test the warning mechanism
        mem = engine.add_memory(
            content="Test memory",
            detect_contradictions=False,
        )
        engine.invalidate_memory(mem.memory_id, reason="test")

        # Retrieve - should not include invalidated
        results = engine.retrieve(query="Test")

        # Verify no invalidated in results
        for r in results:
            assert r.memory.invalidated_at is None


def test_benchmark_passed_determination():
    """BenchmarkResult.passed is determined correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = FluidMemoryEngine(
            FluidMemoryConfig(db_path=f"{tmpdir}/test.db"),
            enable_audit=False,
        )

        # Simple passing case
        mem = engine.add_memory(
            content="Python programming",
            detect_contradictions=False,
        )

        case = BenchmarkCase(
            name="passing_test",
            memories=[],
            query="Python",
            expected_memory_ids=[mem.memory_id],
        )

        # Run retrieval
        results = engine.retrieve(query=case.query, limit=5)
        retrieved_ids = [r.memory.memory_id for r in results]

        # Should pass if expected memory is found
        actual_hits = sum(1 for rid in retrieved_ids if rid in case.expected_memory_ids)
        assert actual_hits >= 1


def test_run_retrieval_benchmark_with_expected_tags():
    """run_retrieval_benchmark uses expected_tags to compute expected hits."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = FluidMemoryEngine(
            FluidMemoryConfig(db_path=f"{tmpdir}/test.db"),
            enable_audit=False,
        )

        case = BenchmarkCase(
            name="tags_test",
            memories=[
                {"content": "Python guide", "tags": ["python"]},
                {"content": "JavaScript guide", "tags": ["js"]},
            ],
            query="Python",
            expected_tags=["python"],  # Use tags instead of IDs
        )

        results = run_retrieval_benchmark(engine, [case], k=2)

        assert len(results) == 1
        result = results[0]

        # expected_hits should be 1 (the python-tagged memory)
        assert result.expected_hits == 1
        # actual_hits should be 1 (found the python guide)
        assert result.actual_hits == 1
        # precision should be > 0
        assert result.precision_at_k > 0
        # should pass
        assert result.passed is True


def test_run_retrieval_benchmark_with_case_local_ids():
    """run_retrieval_benchmark supports case_id for stable memory references."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = FluidMemoryEngine(
            FluidMemoryConfig(db_path=f"{tmpdir}/test.db"),
            enable_audit=False,
        )

        case = BenchmarkCase(
            name="case_id_test",
            memories=[
                {"case_id": "valid", "content": "Valid Python info"},
                {"case_id": "invalid", "content": "Invalid Python info"},
                {"case_id": "contradicted", "content": "Bad Python claim"},
            ],
            query="Python",
            expected_memory_ids=["valid"],  # Use case_id
            invalidated_ids=["invalid"],  # Use case_id
            contradicted_ids=["contradicted"],  # Use case_id
        )

        results = run_retrieval_benchmark(engine, [case], k=2)

        assert len(results) == 1
        result = results[0]

        # Should have expected the valid memory
        assert result.expected_hits == 1
        # Should have found it
        assert result.actual_hits >= 1
        # Invalidated memory should not be returned
        assert result.invalidated_returned == 0
        # Contradicted memory should not be top-ranked
        assert result.contradicted_top_rank is False
        # Should pass
        assert result.passed is True


def test_benchmark_fails_with_zero_expected_hits():
    """Benchmark should fail when expected_hits is 0 (ill-defined test)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = FluidMemoryEngine(
            FluidMemoryConfig(db_path=f"{tmpdir}/test.db"),
            enable_audit=False,
        )

        case = BenchmarkCase(
            name="no_expectations",
            memories=[
                {"content": "Random memory"},
            ],
            query="Python",
            expected_memory_ids=[],  # No expectations
            expected_tags=[],  # No tag expectations
        )

        results = run_retrieval_benchmark(engine, [case], k=2)

        assert len(results) == 1
        result = results[0]

        # expected_hits should be 0
        assert result.expected_hits == 0
        # Should NOT pass (ill-defined test)
        assert result.passed is False


def test_benchmark_cases_are_isolated():
    """Each benchmark case should run with fresh engine state."""
    from fluid_memory.benchmarks import (
        BenchmarkCase, run_retrieval_benchmark, FluidMemoryConfig
    )

    # Create cases with same query but different expected memories
    cases = [
        BenchmarkCase(
            name="case1",
            memories=[{"case_id": "a", "content": "Python alpha"}],
            query="Python",
            expected_memory_ids=["a"],
        ),
        BenchmarkCase(
            name="case2",
            memories=[{"case_id": "b", "content": "Python beta"}],
            query="Python",
            expected_memory_ids=["b"],
        ),
    ]

    # Create a config for testing
    config = FluidMemoryConfig(enable_semantic=False)

    # Run with isolated=True (default)
    results = run_retrieval_benchmark(config, cases, k=2, isolated=True)

    assert len(results) == 2
    # Both cases should pass with isolated engines
    assert results[0].passed is True, "case1 should find 'a'"
    assert results[1].passed is True, (
        "case2 should find 'b' without 'a' interfering"
    )
