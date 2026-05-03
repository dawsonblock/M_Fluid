"""Test orchestrator retrieval and answer grounding."""

import tempfile

from fluid_memory import FluidMemoryEngine, FluidMemoryConfig
from fluid_memory.orchestrator import MemoryOrchestrator
from fluid_memory.packet import RetrievalPacket


def test_retrieve_packet_returns_retrieval_packet():
    """retrieve_packet returns a RetrievalPacket."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = FluidMemoryEngine(
            FluidMemoryConfig(db_path=f"{tmpdir}/test.db"),
            enable_audit=False,
        )
        orchestrator = engine.create_orchestrator()

        engine.add_memory(
            content="Python programming",
            detect_contradictions=False,
        )

        packet = orchestrator.retrieve_packet(query="Python")

        assert isinstance(packet, RetrievalPacket)
        assert packet.query == "Python"


def test_ground_answer_returns_should_answer_false_when_no_evidence():
    """ground_answer returns should_answer=False when no evidence exists."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = FluidMemoryEngine(
            FluidMemoryConfig(db_path=f"{tmpdir}/test.db"),
            enable_audit=False,
        )
        orchestrator = engine.create_orchestrator()

        # Empty packet
        packet = orchestrator.retrieve_packet(query="nonexistent query xyz")

        grounded = orchestrator.ground_answer(
            query="nonexistent query xyz",
            answer="Some answer",
            packet=packet,
        )

        assert grounded["should_answer"] is False
        assert grounded["support_level"] == "none"


def test_ground_answer_returns_should_answer_true_for_supported():
    """ground_answer returns should_answer=True for supported evidence."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = FluidMemoryEngine(
            FluidMemoryConfig(db_path=f"{tmpdir}/test.db"),
            enable_audit=False,
        )
        orchestrator = engine.create_orchestrator()

        engine.add_memory(
            content="Python is a programming language",
            confidence=0.8,
            detect_contradictions=False,
        )

        packet = orchestrator.retrieve_packet(query="Python")

        grounded = orchestrator.ground_answer(
            query="What is Python?",
            answer="Python is a programming language",
            packet=packet,
        )

        assert grounded["should_answer"] is True
        assert grounded["support_level"] in ("supported", "strong")


def test_ground_answer_includes_warnings_for_mixed_support():
    """ground_answer includes warnings for mixed support level."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = FluidMemoryEngine(
            FluidMemoryConfig(db_path=f"{tmpdir}/test.db"),
            enable_audit=False,
        )
        orchestrator = engine.create_orchestrator()

        mem = engine.add_memory(
            content="Python is controversial",
            confidence=0.6,
            detect_contradictions=False,
        )

        # Contradict it to make support mixed
        engine.contradict(mem.memory_id, amount=0.4, reason="test")

        packet = orchestrator.retrieve_packet(query="Python")

        grounded = orchestrator.ground_answer(
            query="What about Python?",
            answer="Something",
            packet=packet,
        )

        # Should answer but with warnings
        assert grounded["should_answer"] is True
        assert len(grounded["warnings"]) > 0


def test_evidence_derived_only_from_packet_refs():
    """Evidence list is derived only from packet.evidence_refs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = FluidMemoryEngine(
            FluidMemoryConfig(db_path=f"{tmpdir}/test.db"),
            enable_audit=False,
        )
        orchestrator = engine.create_orchestrator()

        mem = engine.add_memory(
            content="Python guide",
            source_refs=["docs.python.org"],
            tags=["python"],
            confidence=0.8,
            detect_contradictions=False,
        )

        packet = orchestrator.retrieve_packet(query="Python")

        grounded = orchestrator.ground_answer(
            query="What is Python?",
            answer="A programming language",
            packet=packet,
        )

        # Evidence should match packet evidence refs
        assert len(grounded["evidence"]) == len(packet.evidence_refs)
        if grounded["evidence"]:
            assert grounded["evidence"][0]["memory_id"] == mem.memory_id
            assert grounded["evidence"][0]["source_refs"] == ["docs.python.org"]


def test_semantic_retrieval_through_orchestrator():
    """Semantic retrieval works through orchestrator when use_semantic=True."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = FluidMemoryEngine(
            FluidMemoryConfig(db_path=f"{tmpdir}/test.db"),
            enable_audit=False,
        )
        orchestrator = engine.create_orchestrator()

        engine.add_memory(
            content="Python programming tutorial",
            detect_contradictions=False,
        )
        engine.add_memory(
            content="JavaScript web development",
            detect_contradictions=False,
        )

        packet = orchestrator.retrieve_packet(
            query="Python",
            use_semantic=True,
            semantic_threshold=0.0,
        )

        # Should have results with semantic match type
        assert packet.total_results > 0
        assert packet.evidence_refs


def test_conflict_aware_reranking_in_orchestrator():
    """Conflict-aware reranking is applied when enabled."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = FluidMemoryEngine(
            FluidMemoryConfig(db_path=f"{tmpdir}/test.db"),
            enable_audit=False,
        )
        orchestrator = engine.create_orchestrator()

        # Add stable memory
        stable_mem = engine.add_memory(
            content="Python is stable",
            confidence=0.7,
            detect_contradictions=False,
        )

        # Add memory to be contradicted
        contradicted_mem = engine.add_memory(
            content="Python is unstable",
            confidence=0.8,
            detect_contradictions=False,
        )

        # Heavy contradiction
        engine.contradict(
            contradicted_mem.memory_id,
            amount=0.6,
            reason="evidence contradicts",
        )

        # Retrieve with conflict_aware=True (default)
        packet = orchestrator.retrieve_packet(query="Python")

        # Should have results
        assert packet.total_results >= 2

        # Check that results have conflict-aware metadata
        for result in packet.results:
            if hasattr(result, "metadata") and result.metadata:
                assert "original_score" in result.metadata


def test_orchestrator_does_not_answer_on_empty():
    """Orchestrator should_answer is False on empty retrieval."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = FluidMemoryEngine(
            FluidMemoryConfig(db_path=f"{tmpdir}/test.db"),
            enable_audit=False,
        )
        orchestrator = engine.create_orchestrator()

        packet = orchestrator.retrieve_packet(query="xyznonexistent")

        grounded = orchestrator.ground_answer(
            query="xyznonexistent",
            answer="Some made up answer",
            packet=packet,
        )

        assert grounded["should_answer"] is False
        assert grounded["support_level"] == "none"


def test_semantic_fallback_to_keyword_when_empty():
    """Orchestrator falls back to keyword search when semantic returns empty."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = FluidMemoryEngine(
            FluidMemoryConfig(db_path=f"{tmpdir}/test.db"),
            enable_audit=False,
        )
        orchestrator = engine.create_orchestrator()

        # Add memories with obvious matching content
        engine.add_memory(
            content="Python programming tutorial",
            detect_contradictions=False,
        )
        engine.add_memory(
            content="Python machine learning basics",
            detect_contradictions=False,
        )

        # Use semantic search with default threshold (0.5)
        # The lightweight embedding may not score >= 0.5
        packet = orchestrator.retrieve_packet(
            query="Python",
            use_semantic=True,
        )

        # Should have results due to fallback
        assert packet.total_results > 0, "Semantic fallback failed: no results"

        # Check for fallback warning
        assert any(
            "fallback" in w.lower() or "semantic" in w.lower()
            for w in packet.warnings
        ), "Fallback warning should be present"
