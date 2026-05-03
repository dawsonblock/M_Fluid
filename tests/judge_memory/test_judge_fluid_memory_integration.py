"""
Integration test for Judge + Fluid Memory (Phase 12).

Complete test flow:
1. Configure temp evidence vault
2. Ingest evidence text
3. Store immutable evidence
4. Extract/create claims
5. Attach evidence metadata
6. Touch fluid memory state
7. Retrieve claim
8. Verify source profile affected score
9. Invalidate memory
10. Verify normal retrieval hides invalidated memory
11. Verify admin retrieval shows invalidated memory
12. Create contradiction
13. Verify contradiction changes confidence/volatility/stability/salience
14. Verify audit/events exist
"""

import tempfile
import uuid

import pytest

from judge_memory import JudgeMemoryService, JudgeMemoryConfig
from judge_memory.models import EvidenceRecord, ClaimRecord
from fluid_memory import FluidMemoryEngine, FluidMemoryConfig
from fluid_memory.models import MemoryItem


@pytest.mark.asyncio
async def test_complete_memory_lifecycle():
    """Full integration test from evidence ingest to retrieval and invalidation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Configure temp evidence vault and fluid memory
        judge_config = JudgeMemoryConfig(
            data_dir=tmpdir,
            sqlite_path=f"{tmpdir}/judge.db",
        )
        fluid_config = FluidMemoryConfig(
            db_path=f"{tmpdir}/fluid.db",
        )

        # 2. Initialize services
        judge_service = JudgeMemoryService(judge_config)
        fluid_engine = FluidMemoryEngine(fluid_config)

        # 3. Ingest evidence with source type
        evidence_text = "The Supreme Court ruled in favor of the defendant on all counts."
        evidence = await judge_service.ingest_evidence(
            raw_text=evidence_text,
            source_type="court_record",
            source_title="Supreme Court Ruling 2024",
            source_url="http://courts.example.com/ruling/123",
            jurisdiction="US",
        )

        # Verify evidence stored immutably
        assert evidence.evidence_id is not None
        assert evidence.source_type == "court_record"
        assert evidence.jurisdiction == "US"

        # 4. Create claim linked to evidence
        claim = await judge_service.add_claim(
            evidence_id=evidence.evidence_id,
            claim_text="Defendant was found not guilty on all counts.",
            case_id="CASE-2024-001",
        )

        # Verify claim has evidence metadata
        assert claim.evidence_id == evidence.evidence_id
        assert claim.claim_text == "Defendant was found not guilty on all counts."

        # 5. Search claims - verify evidence metadata included
        results = await judge_service.search("defendant guilty")
        claim_results = [r for r in results if r.result_type == "claim"]
        assert len(claim_results) > 0

        claim_result = claim_results[0]
        # Verify evidence metadata flows through
        assert claim_result.source_type == "court_record"
        assert claim_result.jurisdiction == "US"
        assert "evidence_id" in claim_result.metadata
        assert claim_result.metadata["evidence_id"] == evidence.evidence_id

        # 6. Verify source authority affects scoring
        assert "source_authority" in claim_result.metadata
        assert claim_result.metadata["source_authority"] > 0.8  # court_record has high authority

        # 7. Create fluid memory entry for the claim
        test_id = str(uuid.uuid4())[:8]
        fluid_memory = fluid_engine.add_memory(
            content=f"Case {test_id}: Defendant found not guilty - Supreme Court ruling",
            tags=["court", "ruling", f"case-{test_id}"],
            metadata={
                "claim_id": claim.claim_id,
                "evidence_id": evidence.evidence_id,
                "case_id": "CASE-2024-001",
            },
        )

        # Verify fluid memory created
        assert fluid_memory.memory_id is not None
        assert fluid_memory.confidence == 0.5  # Default confidence

        # 8. Verify initial state
        retrieved = fluid_engine.get_memory(fluid_memory.memory_id)
        assert retrieved.confidence == 0.5
        assert retrieved.volatility == 0.3  # Default volatility
        assert retrieved.stability == 0.5  # Default stability
        assert retrieved.attention_salience == 0.5  # Default
        assert retrieved.contradiction_count == 0

        # 9. Create contradiction
        fluid_engine.contradict(
            memory_id=fluid_memory.memory_id,
            amount=0.3,
            conflicting_evidence_id=evidence.evidence_id,
            reason="New evidence contradicts this finding",
        )

        # 10. Verify contradiction changed multiple state fields
        contradicted = fluid_engine.get_memory(fluid_memory.memory_id)
        assert contradicted.confidence < 0.5  # Lowered
        assert contradicted.volatility > 0.3  # Increased
        assert contradicted.stability < 0.5  # Lowered
        assert contradicted.attention_salience > 0.5  # Increased
        assert contradicted.contradiction_count == 1

        # 11. Verify contradiction event recorded
        events = fluid_engine.storage.get_events(fluid_memory.memory_id)
        contradict_events = [e for e in events if e.event_type.value == "contradicted"]
        assert len(contradict_events) == 1
        contradict_event = contradict_events[0]
        # Event data is in delta_json (not metadata_json)
        assert contradict_event.delta_json.get("reason") == "New evidence contradicts this finding"
        assert contradict_event.delta_json.get("conflicting_evidence_id") == evidence.evidence_id

        # 12. Invalidate memory
        fluid_engine.storage.invalidate(
            fluid_memory.memory_id,
            reason="Overturned by higher court",
        )

        # 13. Verify normal retrieval hides invalidated
        with pytest.raises(Exception):  # MemoryNotFoundError
            fluid_engine.get_memory(fluid_memory.memory_id)

        # Verify retrieval doesn't find it
        results = fluid_engine.retrieve(query="defendant", limit=10)
        assert not any(r.memory.memory_id == fluid_memory.memory_id for r in results)

        # 14. Verify admin retrieval shows invalidated
        invalidated = fluid_engine.storage.get_memory(
            fluid_memory.memory_id,
            include_invalidated=True,
        )
        assert invalidated is not None
        assert invalidated.invalidated_at is not None
        assert invalidated.invalidation_reason == "Overturned by higher court"

        # 15. Verify invalidated memories list
        all_invalidated = fluid_engine.storage.get_invalidated_memories()
        assert any(m.memory_id == fluid_memory.memory_id for m in all_invalidated)

        # Cleanup
        await judge_service.close()


@pytest.mark.asyncio
async def test_evidence_vault_health_check():
    """Test that evidence vault health verification works."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = JudgeMemoryConfig(data_dir=tmpdir)
        service = JudgeMemoryService(config)

        # Vault should be healthy
        health = await service.verify_vault()
        assert health["status"] == "ok"
        assert "type" in health
        assert "message" in health

        await service.close()


@pytest.mark.asyncio
async def test_source_registry_affects_scoring():
    """Test that source trust changes affect scoring."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = JudgeMemoryConfig(data_dir=tmpdir)
        service = JudgeMemoryService(config)

        # Ingest evidence with different source types
        court_evidence = await service.ingest_evidence(
            raw_text="Official court ruling document",
            source_type="court_record",
            source_title="Court Ruling",
        )

        blog_evidence = await service.ingest_evidence(
            raw_text="Blog opinion about the case",
            source_type="blog_social",
            source_title="Legal Blog Post",
        )

        # Add claims
        court_claim = await service.add_claim(
            evidence_id=court_evidence.evidence_id,
            claim_text="Court made a ruling",
        )

        blog_claim = await service.add_claim(
            evidence_id=blog_evidence.evidence_id,
            claim_text="My opinion on the ruling",
        )

        # Search and compare scores
        results = await service.search("ruling")
        claim_results = {r.record_id: r for r in results if r.result_type == "claim"}

        # Both should be found
        assert court_claim.claim_id in claim_results
        assert blog_claim.claim_id in claim_results

        court_result = claim_results[court_claim.claim_id]
        blog_result = claim_results[blog_claim.claim_id]

        # Court should have higher source authority
        assert court_result.metadata["source_authority"] > blog_result.metadata["source_authority"]

        await service.close()


def test_fluid_memory_contradiction_preserves_evidence():
    """Test that contradiction does not delete the underlying evidence/memory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = FluidMemoryConfig(db_path=f"{tmpdir}/fluid.db")
        engine = FluidMemoryEngine(config)

        # Add memory
        memory = engine.add_memory(content="Test memory content")

        # Contradict it
        engine.contradict(
            memory_id=memory.memory_id,
            amount=0.2,
            reason="Testing contradiction",
        )

        # Memory should still exist
        retrieved = engine.get_memory(memory.memory_id)
        assert retrieved.content == "Test memory content"
        assert retrieved.contradiction_count == 1

        # Evidence (memory) not deleted
        all_memories = engine.storage.get_all(limit=10, include_invalidated=False)
        assert any(m.memory_id == memory.memory_id for m in all_memories)
