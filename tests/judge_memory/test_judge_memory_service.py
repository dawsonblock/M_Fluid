"""
Test Judge Memory Service.

Tests the main service interface.
"""

import pytest
import tempfile
from pathlib import Path

from judge_memory import JudgeMemoryService, JudgeMemoryConfig


@pytest.fixture
def temp_service():
    """Create temporary service for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = JudgeMemoryConfig(
            data_dir=Path(tmpdir),
            enable_fluid_memory=False,
        )
        service = JudgeMemoryService(config)
        yield service


@pytest.mark.asyncio
async def test_service_initializes(temp_service):
    """Test service initializes correctly."""
    health = await temp_service.healthcheck()
    assert health["status"] == "ok"
    assert health["fluid_enabled"] is False


@pytest.mark.asyncio
async def test_ingest_evidence_returns_record(temp_service):
    """Test ingest_evidence returns EvidenceRecord."""
    evidence = await temp_service.ingest_evidence(
        raw_text="Test court ruling text.",
        source_type="court_record",
        source_title="Smith v. Jones",
        jurisdiction="US-TX",
    )

    assert evidence.evidence_id is not None
    assert evidence.source_type == "court_record"
    assert evidence.content_hash is not None


@pytest.mark.asyncio
async def test_add_claim_returns_claim_record(temp_service):
    """Test add_claim returns ClaimRecord."""
    # First ingest evidence
    evidence = await temp_service.ingest_evidence(
        raw_text="Test ruling.",
        source_type="court_record",
    )

    # Add claim
    claim = await temp_service.add_claim(
        evidence_id=evidence.evidence_id,
        claim_text="The court ruled for the plaintiff.",
        claim_type="ruling",
        case_id="case_001",
    )

    assert claim.claim_id is not None
    assert claim.evidence_id == evidence.evidence_id
    assert claim.claim_text == "The court ruled for the plaintiff."


@pytest.mark.asyncio
async def test_search_returns_results(temp_service):
    """Test search returns results."""
    # Ingest evidence
    await temp_service.ingest_evidence(
        raw_text="The court ruled in favor of the plaintiff Smith.",
        source_type="court_record",
        source_title="Smith v. Jones",
    )

    # Search
    results = await temp_service.search("plaintiff Smith")

    # Should find the evidence
    assert len(results) > 0
    assert any("plaintiff" in r.summary.lower() for r in results)


@pytest.mark.asyncio
async def test_get_source_packet_returns_profile(temp_service):
    """Test get_source_packet returns source profile."""
    # Ingest evidence
    evidence = await temp_service.ingest_evidence(
        raw_text="Court ruling.",
        source_type="court_record",
        source_title="Test Case",
    )

    # Get source packet
    packet = await temp_service.get_source_packet(evidence.evidence_id)

    assert packet.evidence_id == evidence.evidence_id
    assert packet.source_type == "court_record"
    assert packet.authority is not None
    assert packet.legal_status_label is not None


@pytest.mark.asyncio
async def test_healthcheck_returns_ok(temp_service):
    """Test healthcheck returns ok status."""
    health = await temp_service.healthcheck()

    assert health["status"] == "ok"
    assert "storage" in health
    assert "config" in health
    assert health["fluid_enabled"] is False
