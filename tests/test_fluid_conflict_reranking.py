"""Test conflict-aware reranking."""

import tempfile

from fluid_memory import FluidMemoryEngine, FluidMemoryConfig
from fluid_memory.conflicts import (
    compute_conflict_penalty,
    compute_support_strength,
    rerank_conflict_aware,
)
from fluid_memory.retrieval import RetrievalResult


def test_highly_contradicted_reranked_below_stable():
    """High-score but contradicted memory is reranked below stable lower-score memory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = FluidMemoryEngine(
            FluidMemoryConfig(db_path=f"{tmpdir}/test.db"),
            enable_audit=False,
        )

        # Add a stable memory with moderate score potential
        stable_mem = engine.add_memory(
            content="Python is versatile",
            confidence=0.7,
            detect_contradictions=False,
        )

        # Add a memory that will be heavily contradicted
        contradicted_mem = engine.add_memory(
            content="Python is the fastest language",
            confidence=0.9,
            detect_contradictions=False,
        )

        # Contradict it multiple times to build up penalty
        for _ in range(3):
            engine.contradict(
                contradicted_mem.memory_id,
                amount=0.4,
                reason="benchmarks contradict this",
            )
            contradicted_mem = engine.get_memory(contradicted_mem.memory_id)

        # Get updated stable memory
        stable_mem = engine.get_memory(stable_mem.memory_id)

        # Create results: contradicted has higher raw score but heavy penalty
        results = [
            RetrievalResult(
                memory=contradicted_mem,
                score=0.95,  # Higher raw score
                match_type="text",
            ),
            RetrievalResult(
                memory=stable_mem,
                score=0.65,  # Lower raw score but stable
                match_type="text",
            ),
        ]

        # Rerank conflict-aware
        reranked = rerank_conflict_aware(results)

        # Stable memory should now be first due to lower conflict penalty
        assert reranked[0].memory.memory_id == stable_mem.memory_id
        assert reranked[1].memory.memory_id == contradicted_mem.memory_id


def test_reranking_does_not_drop_results():
    """Conflict-aware reranking does not drop any results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = FluidMemoryEngine(
            FluidMemoryConfig(db_path=f"{tmpdir}/test.db"),
            enable_audit=False,
        )

        memories = []
        for i in range(5):
            mem = engine.add_memory(
                content=f"Memory content {i}",
                detect_contradictions=False,
            )
            memories.append(mem)

        results = [
            RetrievalResult(memory=mem, score=0.5 + i * 0.1, match_type="text")
            for i, mem in enumerate(memories)
        ]

        reranked = rerank_conflict_aware(results)

        assert len(reranked) == len(results)
        # All original IDs should still be present
        original_ids = {r.memory.memory_id for r in results}
        reranked_ids = {r.memory.memory_id for r in reranked}
        assert original_ids == reranked_ids


def test_conflict_penalty_clamped():
    """compute_conflict_penalty is clamped between 0 and 0.8."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = FluidMemoryEngine(
            FluidMemoryConfig(db_path=f"{tmpdir}/test.db"),
            enable_audit=False,
        )

        mem = engine.add_memory(
            content="Test memory",
            detect_contradictions=False,
        )

        # Base case - should have low penalty
        penalty = compute_conflict_penalty(mem)
        assert 0.0 <= penalty <= 0.8

        # Contradict heavily many times
        for i in range(10):
            engine.contradict(mem.memory_id, amount=0.3, reason="test")
            mem = engine.get_memory(mem.memory_id)

        penalty = compute_conflict_penalty(mem)
        assert penalty <= 0.8  # Should be clamped
        assert penalty >= 0.0


def test_support_strength_clamped():
    """compute_support_strength is clamped between 0 and 1."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = FluidMemoryEngine(
            FluidMemoryConfig(db_path=f"{tmpdir}/test.db"),
            enable_audit=False,
        )

        mem = engine.add_memory(
            content="Test memory",
            confidence=1.0,
            detect_contradictions=False,
        )

        strength = compute_support_strength(mem)
        assert 0.0 <= strength <= 1.0

        # Make it very weak
        for i in range(5):
            engine.contradict(mem.memory_id, amount=0.5, reason="test")
            mem = engine.get_memory(mem.memory_id)

        # Also increase volatility using state_delta
        engine.mutate(mem.memory_id, state_delta={"volatility": 0.9})
        mem = engine.get_memory(mem.memory_id)

        strength = compute_support_strength(mem)
        assert 0.0 <= strength <= 1.0


def test_reranking_preserves_determinism():
    """Reranking produces deterministic order for equal adjusted scores."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = FluidMemoryEngine(
            FluidMemoryConfig(db_path=f"{tmpdir}/test.db"),
            enable_audit=False,
        )

        # Add two identical memories (same state)
        mem1 = engine.add_memory(
            content="Memory one",
            confidence=0.5,
            detect_contradictions=False,
        )
        mem2 = engine.add_memory(
            content="Memory two",
            confidence=0.5,
            detect_contradictions=False,
        )

        # Give them same score
        results = [
            RetrievalResult(memory=mem1, score=0.6, match_type="text"),
            RetrievalResult(memory=mem2, score=0.6, match_type="text"),
        ]

        # Run multiple times
        orders = []
        for _ in range(3):
            reranked = rerank_conflict_aware(results)
            order = tuple(r.memory.memory_id for r in reranked)
            orders.append(order)

        # All orders should be the same (deterministic)
        assert orders[0] == orders[1] == orders[2]


def test_original_score_preserved_in_metadata():
    """Original score is preserved in result metadata after reranking."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = FluidMemoryEngine(
            FluidMemoryConfig(db_path=f"{tmpdir}/test.db"),
            enable_audit=False,
        )

        mem = engine.add_memory(
            content="Test memory",
            detect_contradictions=False,
        )

        original_score = 0.8
        results = [
            RetrievalResult(memory=mem, score=original_score, match_type="text"),
        ]

        reranked = rerank_conflict_aware(results)

        # Metadata should have original score and penalty
        assert reranked[0].metadata["original_score"] == original_score
        assert "conflict_penalty" in reranked[0].metadata
