"""
Test Judge Memory Storage.

Tests SQLite storage for evidence, claims, and deduplication.
"""

import pytest
import tempfile
from pathlib import Path
from m_flow.judge_memory.config import JudgeMemoryConfig
from m_flow.judge_memory.storage import JudgeMemoryStorage
from m_flow.judge_memory.models import EvidenceRecord, ClaimRecord


@pytest.fixture
def temp_storage():
    """Create temporary storage for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = JudgeMemoryConfig(
            data_dir=Path(tmpdir),
            enable_fluid_memory=False,
        )
        storage = JudgeMemoryStorage(config)
        yield storage


def test_create_sqlite_db(temp_storage):
    """Test that SQLite DB is created."""
    health = temp_storage.healthcheck()
    assert health["status"] == "ok"
    assert Path(temp_storage.db_path).exists()


def test_store_evidence(temp_storage):
    """Test storing evidence record."""
    evidence = EvidenceRecord(
        evidence_id="ev_test_001",
        source_type="court_record",
        source_title="Test Court Ruling",
        raw_text="This is a test court ruling.",
        content_hash="abc123hash",
        jurisdiction="US-TX",
    )

    result = temp_storage.store_evidence(evidence)
    assert result.evidence_id == "ev_test_001"


def test_duplicate_hash_returns_same_evidence(temp_storage):
    """Test that duplicate content hash returns existing record."""
    evidence1 = EvidenceRecord(
        evidence_id="ev_test_001",
        source_type="court_record",
        raw_text="Duplicate content here.",
        content_hash="dup_hash_123",
    )

    # Store first time
    result1 = temp_storage.store_evidence(evidence1)
    assert result1.evidence_id == "ev_test_001"

    # Try to store with same hash but different ID
    evidence2 = EvidenceRecord(
        evidence_id="ev_test_002",
        source_type="blog_social",
        raw_text="Duplicate content here.",
        content_hash="dup_hash_123",
    )

    # Should return existing record
    result2 = temp_storage.store_evidence(evidence2)
    assert result2.evidence_id == "ev_test_001"  # Returns first one


def test_get_evidence(temp_storage):
    """Test retrieving evidence by ID."""
    evidence = EvidenceRecord(
        evidence_id="ev_test_003",
        source_type="government_data",
        source_title="Government Report",
        raw_text="Government report content.",
        content_hash="gov_hash_456",
    )

    temp_storage.store_evidence(evidence)
    retrieved = temp_storage.get_evidence("ev_test_003")

    assert retrieved is not None
    assert retrieved.evidence_id == "ev_test_003"
    assert retrieved.source_type == "government_data"


def test_store_claim_linked_to_evidence(temp_storage):
    """Test storing claim linked to evidence."""
    # First store evidence
    evidence = EvidenceRecord(
        evidence_id="ev_claim_test",
        source_type="court_record",
        raw_text="Court ruling text.",
        content_hash="claim_ev_hash",
    )
    temp_storage.store_evidence(evidence)

    # Store claim linked to evidence
    claim = ClaimRecord(
        claim_id="cl_test_001",
        evidence_id="ev_claim_test",
        claim_text="The court ruled for the plaintiff.",
        claim_type="ruling",
        confidence=0.9,
    )

    result = temp_storage.store_claim(claim)
    assert result.claim_id == "cl_test_001"
    assert result.evidence_id == "ev_claim_test"


def test_get_claims_by_evidence(temp_storage):
    """Test retrieving claims for evidence."""
    # Setup evidence and claims
    evidence = EvidenceRecord(
        evidence_id="ev_multi_claim",
        source_type="expert_report",
        raw_text="Expert report content.",
        content_hash="multi_claim_hash",
    )
    temp_storage.store_evidence(evidence)

    claim1 = ClaimRecord(
        claim_id="cl_001",
        evidence_id="ev_multi_claim",
        claim_text="First claim.",
    )
    claim2 = ClaimRecord(
        claim_id="cl_002",
        evidence_id="ev_multi_claim",
        claim_text="Second claim.",
    )
    temp_storage.store_claim(claim1)
    temp_storage.store_claim(claim2)

    # Get claims
    claims = temp_storage.get_claims_by_evidence("ev_multi_claim")
    assert len(claims) == 2


def test_claim_storage_without_evidence(temp_storage):
    """Test claim stored without evidence (SQLite FK not enforced)."""
    claim = ClaimRecord(
        claim_id="cl_orphan",
        evidence_id="ev_nonexistent",
        claim_text="Orphan claim.",
    )

    # Should still store (SQLite doesn't enforce FK by default)
    # But our claims manager should check
    temp_storage.store_claim(claim)
    # Verify the evidence doesn't exist
    assert temp_storage.get_evidence("ev_nonexistent") is None
