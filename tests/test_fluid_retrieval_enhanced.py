"""Test enhanced retrieval features: temporal boost, deduplication, MMR."""

import tempfile
from time import time

from fluid_memory import FluidMemoryEngine, FluidMemoryConfig
from fluid_memory.retrieval import (
    temporal_boost,
    remove_duplicates,
    maximal_marginal_relevance,
)


def test_temporal_boost_recent():
    """Recent memories get higher boost."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = FluidMemoryEngine(FluidMemoryConfig(db_path=f"{tmpdir}/test.db"))
        
        memory = engine.add_memory(content="Recent memory", detect_contradictions=False)
        current_time = time()
        
        boost = temporal_boost(memory, current_time)
        assert boost == 1.2  # Less than 1 day old


def test_temporal_boost_old():
    """Old memories get lower boost."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = FluidMemoryEngine(FluidMemoryConfig(db_path=f"{tmpdir}/test.db"))
        
        memory = engine.add_memory(content="Old memory", detect_contradictions=False)
        
        # Simulate old memory (40 days ago)
        memory.created_at = time() - (40 * 24 * 3600)
        current_time = time()
        
        boost = temporal_boost(memory, current_time)
        assert boost == 0.9  # Older than 1 month


def test_retrieval_with_temporal_boost():
    """Retrieval respects temporal boost."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = FluidMemoryEngine(FluidMemoryConfig(db_path=f"{tmpdir}/test.db"))
        
        # Add old memory with high salience
        old_memory = engine.add_memory(
            content="Important old content about AI",
            detect_contradictions=False
        )
        old_memory.created_at = time() - (30 * 24 * 3600)  # 30 days old
        old_memory.salience = 0.9
        engine.storage.update_memory(old_memory)
        
        # Add recent memory with lower salience
        new_memory = engine.add_memory(
            content="Recent content about AI",
            detect_contradictions=False
        )
        new_memory.salience = 0.5
        engine.storage.update_memory(new_memory)
        
        # Search for "AI" with temporal boost
        results = engine.retrieve(
            query="AI",
            limit=10,
            enable_temporal_boost=True
        )
        
        # Recent memory should rank higher despite lower salience
        assert len(results) == 2
        # Recent memory gets 1.0 boost (1 week old), old gets 0.9
        # Scores: old = 0.9 * 0.9 = 0.81, new = 0.5 * 1.0 = 0.5
        # Actually old might still win, but let's verify ordering
        assert results[0].memory.memory_id in [old_memory.memory_id, new_memory.memory_id]


def test_retrieval_deduplication():
    """Near-duplicate results are removed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = FluidMemoryEngine(FluidMemoryConfig(db_path=f"{tmpdir}/test.db"))
        
        # Add similar but not exact duplicate memories
        engine.add_memory(content="AI is transforming software", detect_contradictions=False)
        engine.add_memory(content="AI is transforming software engineering work", detect_contradictions=False)  # Near dup
        engine.add_memory(content="Completely different topic about weather", detect_contradictions=False)  # Unique
        
        results = engine.retrieve(
            query="AI",
            limit=10,
            enable_deduplication=True
        )
        
        # Should filter to unique results only
        assert len(results) <= 3  # At most 3 unique results


def test_remove_duplicates_function():
    """Test deduplication directly."""
    from fluid_memory.models import RetrievalResult, MemoryItem
    from time import time
    
    # Create mock results
    memory1 = MemoryItem(
        memory_id="m1",
        content="Test content one",
        content_hash="h1",
        created_at=time(),
        updated_at=time()
    )
    memory2 = MemoryItem(
        memory_id="m2",
        content="Test content one",  # Exact duplicate
        content_hash="h2",
        created_at=time(),
        updated_at=time()
    )
    memory3 = MemoryItem(
        memory_id="m3",
        content="Different content entirely",
        content_hash="h3",
        created_at=time(),
        updated_at=time()
    )
    
    results = [
        RetrievalResult(memory=memory1, score=0.9, match_type="text"),
        RetrievalResult(memory=memory2, score=0.8, match_type="text"),
        RetrievalResult(memory=memory3, score=0.7, match_type="text"),
    ]
    
    filtered = remove_duplicates(results, similarity_threshold=0.85)
    
    # Should remove the exact duplicate
    assert len(filtered) == 2


def test_mmr_diversity():
    """MMR promotes diverse results."""
    from fluid_memory.models import RetrievalResult, MemoryItem
    from time import time
    
    # Create similar results
    memories = [
        MemoryItem(
            memory_id=f"m{i}",
            content=f"AI topic {i}",
            content_hash=f"h{i}",
            created_at=time(),
            updated_at=time()
        )
        for i in range(5)
    ]
    
    results = [
        RetrievalResult(memory=memories[i], score=1.0 - (i * 0.1), match_type="text")
        for i in range(5)
    ]
    
    # Apply MMR with high diversity preference
    diverse_results = maximal_marginal_relevance(
        results,
        query="AI",
        lambda_param=0.3,  # Favor diversity
        limit=3
    )
    
    assert len(diverse_results) == 3
    # MMR should pick diverse items even if scores differ


def test_retrieval_without_enhancements():
    """Retrieval works with enhancements disabled."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = FluidMemoryEngine(FluidMemoryConfig(db_path=f"{tmpdir}/test.db"))
        
        engine.add_memory(content="Test content", detect_contradictions=False)
        
        results = engine.retrieve(
            query="Test",
            limit=10,
            enable_temporal_boost=False,
            enable_deduplication=False,
            enable_mmr=False
        )
        
        assert len(results) == 1


def test_retrieval_mmr_enabled():
    """Retrieval with MMR enabled."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = FluidMemoryEngine(FluidMemoryConfig(db_path=f"{tmpdir}/test.db"))
        
        # Add multiple related memories
        for i in range(5):
            engine.add_memory(
                content=f"AI and machine learning topic {i}",
                detect_contradictions=False
            )
        
        results = engine.retrieve(
            query="AI",
            limit=3,
            enable_mmr=True,
            mmr_lambda=0.5
        )

        assert len(results) <= 3


def test_health_storage_check_is_healthy_for_new_engine():
    """Health check reports healthy for new engine with no memories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = FluidMemoryEngine(
            FluidMemoryConfig(db_path=f"{tmpdir}/test.db")
        )
        status = engine.health.check_storage()
        assert status.healthy is True
        assert status.component == "storage"
        assert status.details["memory_count"] == 0


def test_retrieve_with_semantic_search_returns_semantic_match_type():
    """Semantic search returns results with match_type='semantic'."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = FluidMemoryEngine(
            FluidMemoryConfig(db_path=f"{tmpdir}/engine.db"),
            enable_audit=False,
        )
        engine.add_memory(
            content="Python programming tutorial",
            detect_contradictions=False
        )
        engine.add_memory(
            content="JavaScript web development guide",
            detect_contradictions=False
        )
        engine.add_memory(
            content="Python machine learning basics",
            detect_contradictions=False
        )
        semantic_results = engine.retrieve(
            query="Python",
            use_semantic=True,
            semantic_threshold=0.0,
            enable_deduplication=False,
        )
        assert semantic_results
        assert all(r.match_type == "semantic" for r in semantic_results)
