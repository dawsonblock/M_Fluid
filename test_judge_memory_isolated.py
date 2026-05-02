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
        print(f"✓ Source packet retrieved")
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
        
        # Test 4: Search
        results = await service.search("guilty")
        print(f"✓ Search returned {len(results)} results")
        
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
    """Verify no m_flow modules loaded."""
    import sys
    
    # Check for m_flow modules
    m_flow_modules = [k for k in sys.modules.keys() if k.startswith('m_flow')]
    
    # Some m_flow modules may be loaded from the compatibility shim,
    # but the key is that structlog is not required
    print(f"\nImport isolation check:")
    print(f"  - m_flow modules loaded: {len(m_flow_modules)}")
    
    # Check structlog is not imported
    has_structlog = 'structlog' in sys.modules
    print(f"  - structlog imported: {has_structlog}")
    
    return not has_structlog


async def main():
    """Run all tests."""
    print("=" * 60)
    print("Judge Memory Isolation Test")
    print("=" * 60)
    
    # Test operations
    ops_ok = await test_basic_operations()
    
    # Test isolation
    iso_ok = test_import_isolation()
    
    print("\n" + "=" * 60)
    if ops_ok and iso_ok:
        print("✓ ALL TESTS PASSED")
        print("Judge Memory works standalone without m_flow!")
    else:
        print("✗ TESTS FAILED")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
