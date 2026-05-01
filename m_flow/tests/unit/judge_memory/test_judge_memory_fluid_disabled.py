"""
Test Judge Memory with Fluid Disabled.

Tests that the service works without fluid memory enabled.
"""

import pytest
import tempfile
from pathlib import Path

from m_flow.judge_memory import JudgeMemoryService, JudgeMemoryConfig


@pytest.fixture
def service_fluid_disabled():
    """Create service with fluid disabled (default)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = JudgeMemoryConfig(
            data_dir=Path(tmpdir),
            enable_fluid_memory=False,  # Default, safe
        )
        service = JudgeMemoryService(config)
        yield service


@pytest.mark.asyncio
async def test_fluid_disabled_by_default(service_fluid_disabled):
    """Test that fluid memory is disabled by default."""
    health = await service_fluid_disabled.healthcheck()
    assert health["fluid_enabled"] is False


@pytest.mark.asyncio
async def test_ingest_does_not_create_fluid_updates(service_fluid_disabled):
    """Test ingest does not touch fluid when disabled."""
    evidence = await service_fluid_disabled.ingest_evidence(
        raw_text="Test evidence.",
        source_type="mainstream_news",
    )

    # Evidence should be stored
    assert evidence.evidence_id is not None

    # Fluid state should not exist (would be None)
    fluid_state = await service_fluid_disabled.fluid.get_state(evidence.evidence_id)
    assert fluid_state is None


@pytest.mark.asyncio
async def test_search_works_without_fluid(service_fluid_disabled):
    """Test search works when fluid is disabled."""
    # Ingest evidence
    await service_fluid_disabled.ingest_evidence(
        raw_text="Important court ruling about plaintiff.",
        source_type="court_record",
    )

    # Search should still work
    results = await service_fluid_disabled.search("court ruling")
    assert len(results) > 0

    # Results should not have fluid scores
    for r in results:
        assert r.fluid_score is None
