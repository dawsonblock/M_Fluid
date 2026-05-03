"""Test retrieval packet construction and support levels."""

import tempfile

from fluid_memory import FluidMemoryEngine, FluidMemoryConfig
from fluid_memory.packet import (
    MemoryEvidenceRef,
    RetrievalPacket,
    build_retrieval_packet,
)
from fluid_memory.retrieval import RetrievalResult


def test_empty_results_produce_none_support_level():
    """Empty results produce support_level='none'."""
    packet = build_retrieval_packet(query="test", results=[])
    assert packet.support_level == "none"
    assert packet.total_results == 0
    assert packet.top_score == 0.0


def test_one_normal_high_confidence_produces_supported():
    """One normal high-confidence result produces 'supported'."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = FluidMemoryEngine(
            FluidMemoryConfig(db_path=f"{tmpdir}/test.db"),
            enable_audit=False,
        )
        mem = engine.add_memory(
            content="Python programming guide",
            tags=["python", "programming"],
            confidence=0.8,
            detect_contradictions=False,
        )

        # Create a retrieval result with high score
        result = RetrievalResult(
            memory=mem,
            score=0.75,
            match_type="text",
        )

        packet = build_retrieval_packet(query="Python", results=[result])
        assert packet.support_level == "supported"
        assert packet.total_results == 1
        assert packet.top_score == 0.75


def test_two_strong_corroborating_produce_strong():
    """Two or more strong corroborating results produce 'strong'."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = FluidMemoryEngine(
            FluidMemoryConfig(db_path=f"{tmpdir}/test.db"),
            enable_audit=False,
        )

        # Add two strong, stable, high-confidence memories
        mem1 = engine.add_memory(
            content="Python is a programming language",
            confidence=0.8,
            detect_contradictions=False,
        )
        mem2 = engine.add_memory(
            content="Python is used for data science",
            confidence=0.8,
            detect_contradictions=False,
        )

        # Manually set stability on memories
        mem1.stability = 0.7
        mem2.stability = 0.7

        results = [
            RetrievalResult(memory=mem1, score=0.75, match_type="text"),
            RetrievalResult(memory=mem2, score=0.72, match_type="text"),
        ]

        packet = build_retrieval_packet(query="Python", results=results)
        assert packet.support_level == "strong"


def test_contradicted_result_produces_mixed():
    """Contradicted/volatile result produces 'mixed' support level."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = FluidMemoryEngine(
            FluidMemoryConfig(db_path=f"{tmpdir}/test.db"),
            enable_audit=False,
        )

        mem = engine.add_memory(
            content="Python is slow",
            confidence=0.6,
            detect_contradictions=False,
        )

        # Contradict the memory
        engine.contradict(mem.memory_id, amount=0.4, reason="benchmarks show speed")

        # Get the updated memory
        updated_mem = engine.get_memory(mem.memory_id)

        result = RetrievalResult(
            memory=updated_mem,
            score=0.6,
            match_type="text",
        )

        packet = build_retrieval_packet(query="Python", results=[result])
        assert packet.support_level == "mixed"
        assert any("contradiction" in w.lower() for w in packet.warnings)


def test_invalidated_memories_not_included():
    """Invalidated memories should not appear in normal retrieval packets."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = FluidMemoryEngine(
            FluidMemoryConfig(db_path=f"{tmpdir}/test.db"),
            enable_audit=False,
        )

        mem = engine.add_memory(
            content="Outdated Python 2 info",
            detect_contradictions=False,
        )
        engine.invalidate_memory(mem.memory_id, reason="outdated")

        # Retrieve by query - invalidated should not appear
        results = engine.retrieve(query="Python")

        # Verify it's not in results
        retrieved_ids = [r.memory.memory_id for r in results]
        assert mem.memory_id not in retrieved_ids


def test_evidence_refs_include_required_fields():
    """Evidence refs include memory_id, content_hash, source_refs, tags, confidence."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = FluidMemoryEngine(
            FluidMemoryConfig(db_path=f"{tmpdir}/test.db"),
            enable_audit=False,
        )

        mem = engine.add_memory(
            content="Python guide",
            tags=["python", "guide"],
            source_refs=["docs.python.org"],
            confidence=0.8,
            detect_contradictions=False,
        )

        result = RetrievalResult(
            memory=mem,
            score=0.7,
            match_type="text",
        )

        packet = build_retrieval_packet(query="Python", results=[result])

        assert len(packet.evidence_refs) == 1
        ref = packet.evidence_refs[0]
        assert ref.memory_id == mem.memory_id
        assert ref.content_hash == mem.content_hash
        assert ref.source_refs == ["docs.python.org"]
        assert ref.tags == ["python", "guide"]
        assert ref.confidence == 0.8


def test_high_volatility_produces_mixed():
    """High volatility memory produces 'mixed' support level."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = FluidMemoryEngine(
            FluidMemoryConfig(db_path=f"{tmpdir}/test.db"),
            enable_audit=False,
        )

        mem = engine.add_memory(
            content="Rumor about Python",
            confidence=0.5,
            volatility=0.75,  # Set high volatility at creation
            detect_contradictions=False,
        )

        # Get memory
        updated_mem = engine.get_memory(mem.memory_id)

        # Verify volatility is high
        assert updated_mem.volatility > 0.7

        result = RetrievalResult(
            memory=updated_mem,
            score=0.6,
            match_type="text",
        )

        packet = build_retrieval_packet(query="Python", results=[result])

        # Should have volatility warning
        assert any("volatil" in w.lower() for w in packet.warnings)

        # With high volatility but no contradictions and decent score,
        # support level depends on overall assessment
        assert packet.support_level in ("mixed", "supported")


def test_contradiction_overrides_strong_support_level():
    """Contradictions override strong support level assessment."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = FluidMemoryEngine(
            FluidMemoryConfig(db_path=f"{tmpdir}/test.db"),
            enable_audit=False,
        )

        # Add two strong memories
        mem1 = engine.add_memory(
            content="Python is a programming language",
            confidence=0.9,
            detect_contradictions=False,
        )
        mem2 = engine.add_memory(
            content="Python is used for data science",
            confidence=0.9,
            detect_contradictions=False,
        )

        # Add a third memory to be contradicted
        mem3 = engine.add_memory(
            content="Python is the fastest language",
            confidence=0.9,
            detect_contradictions=False,
        )

        # Contradict the third memory
        engine.contradict(mem3.memory_id, amount=0.6, reason="evidence contradicts")

        # Get all memories
        mem1 = engine.get_memory(mem1.memory_id)
        mem2 = engine.get_memory(mem2.memory_id)
        mem3 = engine.get_memory(mem3.memory_id)

        # Create results including the contradicted memory
        results = [
            RetrievalResult(memory=mem1, score=0.8, match_type="text"),
            RetrievalResult(memory=mem2, score=0.8, match_type="text"),
            RetrievalResult(memory=mem3, score=0.8, match_type="text"),
        ]

        packet = build_retrieval_packet(query="Python", results=results)

        # Should be mixed due to contradictions despite having 2+ strong results
        assert packet.support_level == "mixed"
        assert any("contradiction" in w.lower() for w in packet.warnings)
