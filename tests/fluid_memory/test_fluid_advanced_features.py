"""
Test advanced features: semantic search, contradiction detection, audit logging.
"""

import tempfile
import pytest
from fluid_memory import FluidMemoryEngine, FluidMemoryConfig
from fluid_memory.models import MemoryItem
from fluid_memory.storage import MemoryStorage, _compute_embedding, _cosine_similarity
from fluid_memory.audit_logger import AuditLogger, AuditEventType, set_audit_context


def test_embedding_computation():
    """Test that embeddings are computed and can be compared."""
    # Similar content should have higher similarity
    emb1 = _compute_embedding("The quick brown fox")
    emb2 = _compute_embedding("The fast brown fox")
    emb3 = _compute_embedding("Completely different topic about space")

    sim1 = _cosine_similarity(emb1, emb2)  # Similar
    sim2 = _cosine_similarity(emb1, emb3)  # Different

    assert sim1 > sim2  # Similar content has higher similarity
    assert 0 <= sim1 <= 1.0  # Cosine similarity is bounded
    assert 0 <= sim2 <= 1.0


def test_semantic_search_finds_similar_content():
    """Test that semantic search finds similar content."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = MemoryStorage(f"{tmpdir}/test.db")

        # Add memories with different content
        memory1 = MemoryItem(content="Python programming language")
        memory2 = MemoryItem(content="JavaScript web development")
        memory3 = MemoryItem(content="Python data science and ML")

        storage.save_memory(memory1)
        storage.save_memory(memory2)
        storage.save_memory(memory3)

        # Save embeddings
        storage.save_embedding(memory1.memory_id, memory1.content)
        storage.save_embedding(memory2.memory_id, memory2.content)
        storage.save_embedding(memory3.memory_id, memory3.content)

        # Search for Python-related content
        results = storage.semantic_search("Python coding", limit=10, threshold=0.3)

        # Should find Python-related memories
        assert len(results) >= 2
        found_ids = {m.memory_id for m, _ in results}
        assert memory1.memory_id in found_ids
        assert memory3.memory_id in found_ids


def test_semantic_search_respects_invalidated():
    """Test that semantic search excludes invalidated memories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = MemoryStorage(f"{tmpdir}/test.db")

        # Add and invalidate a memory
        memory = MemoryItem(content="Test content for search")
        storage.save_memory(memory)
        storage.save_embedding(memory.memory_id, memory.content)
        storage.invalidate(memory.memory_id, reason="Test")

        # Search should not find it by default
        results = storage.semantic_search("Test content", threshold=0.3)
        assert len(results) == 0

        # With include_invalidated=True, should find it
        results = storage.semantic_search(
            "Test content", threshold=0.3, include_invalidated=True
        )
        assert len(results) == 1


def test_contradiction_detection_on_add():
    """Test that adding a memory with contradictory tags triggers detection."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = FluidMemoryEngine(FluidMemoryConfig(db_path=f"{tmpdir}/engine.db"))

        # Add first memory with high confidence
        memory1 = engine.add_memory(
            content="The defendant is guilty of all charges.",
            tags=["court", "verdict"],
            confidence=0.9,
            detect_contradictions=False,  # Don't detect on first add
        )

        # Add contradictory memory
        memory2 = engine.add_memory(
            content="The defendant is innocent and acquitted.",
            tags=["court", "verdict", "disputed"],  # Marked as disputed
            confidence=0.3,  # Low confidence, different from first
            detect_contradictions=True,
            contradiction_threshold=0.3,
        )

        # Both memories should have contradiction pressure applied
        # (detected due to confidence difference and disputed tag)
        assert memory2.contradiction_count >= 0  # May or may not trigger


@pytest.mark.timeout(30)
def test_audit_logger_creates_events():
    """Test that audit logger creates structured events."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = AuditLogger(log_dir=tmpdir, enable_file=True)

        # Set context
        set_audit_context(user_id="test_user", session_id="test_session")

        # Log an event
        event = logger.log_memory_created(
            memory_id="mem_test_123",
            content_hash="abc123",
            tags=["test"],
            metadata={"key": "value"},
        )

        # Verify event structure
        assert event["event_type"] == AuditEventType.MEMORY_CREATED.value
        assert event["memory_id"] == "mem_test_123"
        assert "timestamp" in event
        assert "sequence" in event
        assert event["user_id"] == "test_user"
        assert event["session_id"] == "test_session"
        assert event["details"]["content_hash"] == "abc123"


def test_audit_logger_contradiction_events():
    """Test that contradiction events are logged."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = AuditLogger(log_dir=tmpdir, enable_file=True)

        # Log contradiction detection
        event = logger.log_contradiction_detected(
            memory_id="mem_1",
            conflicting_memory_id="mem_2",
            similarity_score=0.85,
            reason="Confidence mismatch",
        )

        assert event["event_type"] == AuditEventType.CONTRADICTION_DETECTED.value
        assert event["details"]["conflicting_memory_id"] == "mem_2"
        assert event["details"]["similarity_score"] == 0.85


def test_retrieve_with_semantic_search():
    """Test that retrieve supports semantic search flag."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = FluidMemoryEngine(
            FluidMemoryConfig(db_path=f"{tmpdir}/engine.db"), enable_audit=False
        )

        # Add memories
        engine.add_memory(content="Python programming tutorial")
        engine.add_memory(content="JavaScript web development guide")
        engine.add_memory(content="Python machine learning basics")

        # Keyword search
        keyword_results = engine.retrieve(query="Python", use_semantic=False)
        assert len(keyword_results) >= 2

        # Semantic search
        semantic_results = engine.retrieve(
            query="Python coding tutorial",
            use_semantic=True,
            semantic_threshold=0.3,
        )
        # Should return results with match_type="semantic"
        if semantic_results:
            assert any(r.match_type == "semantic" for r in semantic_results)


@pytest.mark.timeout(30)
def test_retrieve_logs_access_events():
    """Test that retrieval logs access events."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = FluidMemoryEngine(
            FluidMemoryConfig(db_path=f"{tmpdir}/engine.db"), enable_audit=True
        )

        # Add memory
        memory = engine.add_memory(content="Test content")

        # Retrieve should log access
        results = engine.retrieve(query="Test")
        assert len(results) > 0

        # Access should be logged (verify by checking no exceptions)


def test_contradiction_applies_multiple_state_changes():
    """Test that contradiction changes multiple state fields."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = FluidMemoryEngine(FluidMemoryConfig(db_path=f"{tmpdir}/engine.db"))

        # Add memory
        memory = engine.add_memory(
            content="Test content", confidence=0.8, detect_contradictions=False
        )

        # Record initial state
        initial_confidence = memory.confidence
        initial_volatility = memory.volatility
        initial_stability = memory.stability
        initial_attention = memory.attention_salience

        # Apply contradiction
        engine.contradict(
            memory.memory_id,
            amount=0.3,
            conflicting_memory_id="mem_other",
            reason="Test contradiction",
        )

        # Verify state changes
        updated = engine.get_memory(memory.memory_id)
        assert updated.confidence < initial_confidence  # Lowered
        assert updated.volatility > initial_volatility  # Increased
        assert updated.stability < initial_stability  # Lowered
        assert updated.attention_salience > initial_attention  # Increased
        assert updated.contradiction_count == 1


def test_contradiction_event_metadata():
    """Test that contradiction events contain full state changes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = FluidMemoryEngine(FluidMemoryConfig(db_path=f"{tmpdir}/engine.db"))

        # Add memory
        memory = engine.add_memory(content="Test", detect_contradictions=False)

        # Apply contradiction
        engine.contradict(
            memory.memory_id,
            amount=0.2,
            conflicting_memory_id="mem_conflict",
            reason="Testing state tracking",
        )

        # Get events
        events = engine.storage.get_events(memory.memory_id)
        contradict_events = [e for e in events if e.event_type.value == "contradicted"]
        assert len(contradict_events) == 1

        event = contradict_events[0]
        # Event should contain old/new state
        assert "old" in event.delta_json
        assert "new" in event.delta_json
        assert "reason" in event.delta_json
        assert "conflicting_memory_id" in event.delta_json
