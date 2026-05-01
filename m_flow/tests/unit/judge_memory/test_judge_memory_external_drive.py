"""
Test Judge Memory External Drive Support.

Tests that evidence can be stored on external/temporary paths.
"""

import pytest
import tempfile
from pathlib import Path

from m_flow.judge_memory import JudgeMemoryService, JudgeMemoryConfig


@pytest.mark.asyncio
async def test_external_style_path():
    """Test config can point to external-style temporary path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Simulate external drive path
        external_path = Path(tmpdir) / "JudgeMemory"

        config = JudgeMemoryConfig(
            data_dir=external_path,
            evidence_dir=external_path / "evidence",
            sqlite_path=external_path / "judge_memory.sqlite",
            enable_fluid_memory=False,
        )

        service = JudgeMemoryService(config)

        # Ingest evidence
        evidence = await service.ingest_evidence(
            raw_text="External drive test evidence.",
            source_type="government_data",
        )

        # Evidence file should be created on "external" path
        assert Path(evidence.storage_path).exists()
        assert external_path in Path(evidence.storage_path).parents


@pytest.mark.asyncio
async def test_existing_evidence_not_overwritten():
    """Test that existing evidence files are not overwritten."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = JudgeMemoryConfig(
            data_dir=Path(tmpdir),
            enable_fluid_memory=False,
        )

        service = JudgeMemoryService(config)

        # Ingest evidence first time
        evidence1 = await service.ingest_evidence(
            raw_text="Original evidence content.",
            source_type="court_record",
        )

        storage_path = evidence1.storage_path
        first_modified = Path(storage_path).stat().st_mtime

        # Try to ingest same content again
        evidence2 = await service.ingest_evidence(
            raw_text="Original evidence content.",  # Same content
            source_type="court_record",
        )

        # Should return same record, not overwrite
        assert evidence2.evidence_id == evidence1.evidence_id

        # File should not have been modified
        second_modified = Path(storage_path).stat().st_mtime
        assert second_modified == first_modified
