"""
Test Judge Memory Storage.

Tests SQLite storage for evidence, claims, and deduplication.
"""

import pytest
import tempfile
from pathlib import Path
from judge_memory.config import JudgeMemoryConfig
from judge_memory.storage import JudgeMemoryStorage
from judge_memory.models import EvidenceRecord, ClaimRecord


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


def test_store_claim_rejects_missing_evidence(temp_storage):
    """Raw storage should reject claims for non-existent evidence."""
    claim = ClaimRecord(
        claim_id="cl_orphan",
        evidence_id="ev_nonexistent",
        claim_text="Orphan claim.",
    )

    # Should reject orphan claims at storage level
    from judge_memory.exceptions import EvidenceNotFoundError
    with pytest.raises(EvidenceNotFoundError) as exc_info:
        temp_storage.store_claim(claim)
    assert "ev_nonexistent" in str(exc_info.value)
    # Verify the evidence doesn't exist
    assert temp_storage.get_evidence("ev_nonexistent") is None


def test_fts_search_sanitizes_special_characters(temp_storage):
    """FTS queries with special chars should not crash."""
    # Store some evidence first
    evidence = EvidenceRecord(
        evidence_id="ev_fts_test",
        source_type="court_record",
        source_title="Court Ruling on Python",
        raw_text="Python programming case.",
        content_hash="fts_test_hash",
    )
    temp_storage.store_evidence(evidence)

    # Search with dangerous characters that would break FTS5
    dangerous_queries = [
        'test * ( ) [ ]',
        'python " quote',
        'test - + : ~',
        'case @ # $ %',
        'ruling ^ & = < >',
    ]

    for query in dangerous_queries:
        # Should not raise sqlite3.OperationalError
        results = temp_storage.search_evidence_fts(query, limit=5)
        # Should return list (may be empty if no match)
        assert isinstance(results, list)


def test_claims_fts_sanitizes_special_characters(temp_storage):
    """FTS claims search with special chars should not crash."""
    # Store evidence and claim
    evidence = EvidenceRecord(
        evidence_id="ev_claims_fts",
        source_type="court_record",
        raw_text="Court ruling text.",
        content_hash="claims_fts_hash",
    )
    temp_storage.store_evidence(evidence)

    claim = ClaimRecord(
        claim_id="cl_fts_test",
        evidence_id="ev_claims_fts",
        claim_text="The court ruled for the plaintiff.",
    )
    temp_storage.store_claim(claim)

    # Search with dangerous characters
    dangerous_queries = [
        'ruling * ( )',
        'court " quote',
        'test - + : ~',
    ]

    for query in dangerous_queries:
        # Should not raise sqlite3.OperationalError
        results = temp_storage.search_claims_fts(query, limit=5)
        assert isinstance(results, list)


def test_source_profile_uses_hardcoded_fallback(temp_storage):
    """Source profiles should work without m_flow registry."""
    from judge_memory.fluid_adapter import FluidMemoryAdapter
    from judge_memory.config import JudgeMemoryConfig

    # Create adapter without m_flow source registry
    config = JudgeMemoryConfig(enable_fluid_memory=False)
    adapter = FluidMemoryAdapter(config, source_registry=None)

    # Known source types should return correct profiles
    court_profile = adapter.get_source_profile("court_record")
    # Court records have high authority
    assert court_profile["authority"] > 0.8
    assert court_profile["legal_status_label"] == "primary_authority"

    # Unknown source types should return "unknown" profile
    unknown_profile = adapter.get_source_profile("unknown_type")
    assert unknown_profile["authority"] < 0.6  # Unknown has lower authority
    assert unknown_profile["legal_status_label"] == "unverified"
