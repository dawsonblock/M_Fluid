"""Test Judge Memory isolated package.

This test verifies that judge_memory works without m_flow dependencies.
"""

import asyncio
import tempfile
from pathlib import Path

from judge_memory import (
    JudgeMemoryService,
    JudgeMemoryConfig,
    EvidenceRecord,
    ClaimRecord,
)


async def test_basic_operations():
    """Test basic evidence and claim operations."""
    
    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as tmpdir:
        # Configure service
        config = JudgeMemoryConfig(
            data_dir=tmpdir,
            enable_fluid_memory=False,  # Ensure no m_flow dependency
        )
        
        # Create service
        service = JudgeMemoryService(config)
        print(f"✓ Service created with data_dir: {tmpdir}")
        
        # Test 1: Ingest evidence
        evidence = await service.ingest_evidence(
            raw_text="The court finds the defendant guilty of all charges.",
            source_type="court_record",
            source_title="State v. Smith",
            jurisdiction="US-TX",
            metadata={"case_number": "2024-001"},
        )
        print(f"✓ Evidence ingested: {evidence.evidence_id}")
        print(f"  - Content hash: {evidence.content_hash[:16]}...")
        print(f"  - Source type: {evidence.source_type}")
        
        # Test 2: Get source packet
        packet = await service.get_source_packet(evidence.evidence_id)
        print("✓ Source packet retrieved")
        print(f"  - Authority: {packet.authority}")
        print(f"  - Legal status: {packet.legal_status_label}")
        
        # Test 3: Add claim
        claim = await service.add_claim(
            evidence_id=evidence.evidence_id,
            claim_text="Defendant found guilty on all counts",
            claim_type="ruling",
            case_id="case_2024_001",
            confidence=0.9,
            tags=["guilty", "verdict"],
        )
        print(f"✓ Claim added: {claim.claim_id}")
        print(f"  - Status: {claim.status}")
        print(f"  - Confidence: {claim.confidence}")
        
        # Test 4: Search evidence by content
        results = await service.search("guilty")
        print(f"✓ Search returned {len(results)} results")
        assert len(results) > 0, "Search should find evidence by content"
        print(f"  - Found: {results[0].result_type} '{results[0].title}'")
        
        # Test 5: Review claim
        reviewed = await service.review_claim(
            claim_id=claim.claim_id,
            status="confirmed",
            reviewed_by="judge_001",
            notes="Verified against court transcript",
        )
        print(f"✓ Claim reviewed: {reviewed.status}")
        
        # Test 6: Close service
        await service.close()
        print("✓ Service closed cleanly")
        
        return True


def test_import_isolation():
    """Verify structlog is not required by judge_memory."""
    import sys

    has_structlog = 'structlog' in sys.modules
    assert not has_structlog, (
        "structlog should not be imported by judge_memory, "
        "but it was found in sys.modules"
    )


async def test_orphaned_claims_blocked():
    """Test that claims cannot be created without evidence."""
    import tempfile
    import pytest
    from judge_memory import JudgeMemoryService, JudgeMemoryConfig
    from judge_memory.exceptions import EvidenceNotFoundError

    with tempfile.TemporaryDirectory() as tmpdir:
        config = JudgeMemoryConfig(data_dir=tmpdir)
        service = JudgeMemoryService(config)

        try:
            with pytest.raises(EvidenceNotFoundError):
                await service.add_claim(
                    evidence_id="nonexistent_evidence",
                    claim_text="This should fail",
                )
        finally:
            await service.close()


async def test_claim_search():
    """Test searching claims by text."""
    import tempfile
    from judge_memory import JudgeMemoryService, JudgeMemoryConfig

    with tempfile.TemporaryDirectory() as tmpdir:
        config = JudgeMemoryConfig(data_dir=tmpdir)
        service = JudgeMemoryService(config)

        # Create evidence and claim
        evidence = await service.ingest_evidence(
            raw_text="Court proceedings transcript",
            source_type="court_record",
        )
        await service.add_claim(
            evidence_id=evidence.evidence_id,
            claim_text="The defendant breached contract obligations",
            case_id="case_001",
        )

        # Search for claim
        results = await service.search("breached contract")
        claim_results = [r for r in results if r.result_type == "claim"]

        assert len(claim_results) > 0, (
            f"Expected at least one claim result, got {len(claim_results)}"
        )

        await service.close()


async def main():
    """Run all tests."""
    print("=" * 60)
    print("Judge Memory Production-Ready Test")
    print("=" * 60)

    results = {}

    for name, coro in [
        ("Basic operations", test_basic_operations()),
        ("Orphan claims blocked", test_orphaned_claims_blocked()),
        ("Claim search", test_claim_search()),
    ]:
        try:
            await coro
            results[name] = True
            print(f"✓ {name}")
        except Exception as exc:
            results[name] = False
            print(f"✗ {name}: {exc}")

    for name, fn in [("Import isolation", test_import_isolation)]:
        try:
            fn()
            results[name] = True
            print(f"✓ {name}")
        except Exception as exc:
            results[name] = False
            print(f"✗ {name}: {exc}")

    print("\n" + "=" * 60)
    all_passed = all(results.values())
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("Judge Memory is production-ready!")
    else:
        print("✗ SOME TESTS FAILED")
        for name, ok in results.items():
            print(f"  - {name}: {'✓' if ok else '✗'}")
    print("=" * 60)
    return all_passed


if __name__ == "__main__":
    asyncio.run(main())
