"""
Test Judge Memory Search improvements for Phase 1.

- Claim search returns evidence metadata
- FTS input sanitization prevents crashes
- Search method is identifiable in metadata
"""

import asyncio
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from judge_memory import JudgeMemoryService, JudgeMemoryConfig
from judge_memory.storage import _sanitize_fts_query, JudgeMemoryStorage
from judge_memory.models import EvidenceRecord, ClaimRecord


def test_sanitize_fts_query_escapes_special_chars():
    """Test that FTS special characters are escaped/removed."""
    # FTS5 special chars: " * ( ) - + : ~ . / \ @ # $ % ^ & = < >
    test_cases = [
        ('hello "world"', 'hello world'),
        ('test*query', 'test query'),
        ('(parentheses)', 'parentheses'),
        ('dash-query', 'dash query'),
        ('plus+query', 'plus query'),
        ('colon:query', 'colon query'),
        ('tilde~query', 'tilde query'),
        ('dot.query', 'dot query'),
        ('slash/query', 'slash query'),
        ('back\\slash', 'back slash'),
        ('at@query', 'at query'),
        ('hash#query', 'hash query'),
        ('dollar$query', 'dollar query'),
        ('percent%query', 'percent query'),
        ('caret^query', 'caret query'),
        ('ampersand&query', 'ampersand query'),
        ('equals=query', 'equals query'),
        ('less<query', 'less query'),
        ('greater>query', 'greater query'),
        # Multiple spaces collapsed
        ('hello   world', 'hello world'),
        # Empty after sanitization
        ('""', ''),
        ('', ''),
    ]

    for input_query, expected in test_cases:
        result = _sanitize_fts_query(input_query)
        assert result == expected, f"Failed for input: {input_query!r}"


def test_sanitize_fts_query_preserves_content():
    """Test that meaningful content is preserved."""
    # Normal text should pass through
    assert _sanitize_fts_query("court bail hearing") == "court bail hearing"
    assert _sanitize_fts_query("Saskatoon court decision") == "Saskatoon court decision"
    # Numbers preserved
    assert _sanitize_fts_query("Case 2024-001") == "Case 2024 001"


@pytest.mark.asyncio
async def test_claim_search_returns_evidence_metadata():
    """Test that claim search joins with evidence and returns source metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = JudgeMemoryConfig(data_dir=tmpdir)
        service = JudgeMemoryService(config)

        # Ingest evidence with specific source type
        evidence_text = "The court granted bail to the defendant on Monday."
        evidence = await service.ingest_evidence(
            raw_text=evidence_text,
            source_type="court_record",
            source_title="Bail Hearing Decision",
            source_url="http://courts.example.com/case/123",
            jurisdiction="SK",
        )

        # Add a claim linked to the evidence
        claim = await service.add_claim(
            evidence_id=evidence.evidence_id,
            claim_text="Bail was granted by the Saskatoon court on Monday.",
            case_id="CASE-2024-001",
        )

        # Search for the claim
        results = await service.search("Saskatoon bail granted")

        # Find the claim result
        claim_results = [r for r in results if r.result_type == "claim"]
        assert len(claim_results) > 0, "Should find claim in search results"

        claim_result = claim_results[0]

        # Verify evidence metadata is present
        assert claim_result.source_type == "court_record", \
            f"Expected source_type='court_record', got {claim_result.source_type!r}"
        assert claim_result.jurisdiction == "SK", \
            f"Expected jurisdiction='SK', got {claim_result.jurisdiction!r}"

        # Verify metadata includes evidence info
        assert "evidence_id" in claim_result.metadata
        assert claim_result.metadata["evidence_id"] == evidence.evidence_id
        assert "evidence_title" in claim_result.metadata
        assert claim_result.metadata["evidence_title"] == "Bail Hearing Decision"
        assert "source_authority" in claim_result.metadata
        assert claim_result.metadata["source_authority"] == 0.9  # court_record authority

        await service.close()


@pytest.mark.asyncio
async def test_claim_search_source_type_affects_scoring():
    """Test that source authority affects claim search scoring."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = JudgeMemoryConfig(data_dir=tmpdir)
        service = JudgeMemoryService(config)

        # Ingest evidence with different source types
        court_evidence = await service.ingest_evidence(
            raw_text="The Supreme Court ruled on the matter.",
            source_type="court_record",
            source_title="Supreme Court Ruling",
            jurisdiction="CA",
        )

        blog_evidence = await service.ingest_evidence(
            raw_text="A blogger opinion about the ruling.",
            source_type="blog_social",
            source_title="My Legal Blog",
            jurisdiction="CA",
        )

        # Add similar claims to both
        court_claim = await service.add_claim(
            evidence_id=court_evidence.evidence_id,
            claim_text="The ruling established important precedent.",
        )

        blog_claim = await service.add_claim(
            evidence_id=blog_evidence.evidence_id,
            claim_text="This ruling sets a precedent for future cases.",
        )

        # Search for precedent-related claim
        results = await service.search("precedent ruling")

        # Filter to claim results
        claim_results = [r for r in results if r.result_type == "claim"]

        # Both claims should be found
        assert len(claim_results) >= 2

        # Court claim should have higher authority score
        court_result = next((r for r in claim_results if r.record_id == court_claim.claim_id), None)
        blog_result = next((r for r in claim_results if r.record_id == blog_claim.claim_id), None)

        assert court_result is not None, "Court claim should be in results"
        assert blog_result is not None, "Blog claim should be in results"

        # Court should have higher source authority
        assert court_result.metadata["source_authority"] > blog_result.metadata["source_authority"]

        await service.close()


@pytest.mark.asyncio
async def test_malformed_fts_input_does_not_crash():
    """Test that FTS input with special characters doesn't crash search."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = JudgeMemoryConfig(data_dir=tmpdir)
        service = JudgeMemoryService(config)

        # Ingest evidence
        evidence = await service.ingest_evidence(
            raw_text="Court decision regarding bail proceedings.",
            source_type="court_record",
            source_title="Decision",
        )

        await service.add_claim(
            evidence_id=evidence.evidence_id,
            claim_text="Bail was discussed in court.",
        )

        # Search with various malformed inputs that could break FTS5
        malformed_queries = [
            '"',  # Unbalanced quote
            '(',  # Unbalanced paren
            ')',  # Unbalanced paren
            '*',  # Wildcard
            '-',  # Minus
            '+',  # Plus
            ':',  # Colon
            '~',  # Tilde
            '.',  # Dot
            '/',  # Slash
            '\\',  # Backslash
            '@',  # At
            '#',  # Hash
            '$',  # Dollar
            '%',  # Percent
            '^',  # Caret
            '&',  # Ampersand
            '=',  # Equals
            '<',  # Less than
            '>',  # Greater than
            '"quoted"',  # Quotes around term
            'term1 -term2',  # Minus operator
            'term1 +term2',  # Plus operator
        ]

        for query in malformed_queries:
            # Should not raise an exception
            try:
                results = await service.search(query)
                # Result may be empty but should not crash
                assert isinstance(results, list)
            except Exception as e:
                pytest.fail(f"Query {query!r} raised exception: {e}")

        await service.close()


@pytest.mark.asyncio
async def test_search_method_identifiable_in_metadata():
    """Test that search results indicate which method was used."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = JudgeMemoryConfig(data_dir=tmpdir)
        service = JudgeMemoryService(config)

        # Ingest and create claim
        evidence = await service.ingest_evidence(
            raw_text="Court proceedings and decisions.",
            source_type="court_record",
            source_title="Proceedings",
        )

        await service.add_claim(
            evidence_id=evidence.evidence_id,
            claim_text="The court made a decision.",
        )

        # Search with query (should use FTS5 if available)
        results_with_query = await service.search("court decision")

        # Check that FTS method is indicated
        for result in results_with_query:
            if result.result_type == "claim":
                assert "search_method" in result.metadata
                # Should be fts5 or fallback_like
                assert result.metadata["search_method"] in ["fts5", "fallback_like"]

        await service.close()


@pytest.mark.asyncio
async def test_claim_result_never_returns_none_source_type():
    """Test that claim search never returns source_type=None when evidence has source."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = JudgeMemoryConfig(data_dir=tmpdir)
        service = JudgeMemoryService(config)

        # Ingest evidence with explicit source type
        evidence = await service.ingest_evidence(
            raw_text="Official court document content.",
            source_type="government_data",
            source_title="Gov Document",
            jurisdiction="ON",
        )

        # Add claim
        await service.add_claim(
            evidence_id=evidence.evidence_id,
            claim_text="Government issued this document.",
        )

        # Search
        results = await service.search("government document")

        # Check claim results
        for result in results:
            if result.result_type == "claim":
                # source_type should NOT be None
                assert result.source_type is not None, \
                    "Claim result should have source_type from evidence"
                assert result.source_type == "government_data", \
                    f"Expected 'government_data', got {result.source_type!r}"
                # jurisdiction should also be preserved
                assert result.jurisdiction == "ON", \
                    f"Expected jurisdiction 'ON', got {result.jurisdiction!r}"

        await service.close()


def test_storage_search_claims_with_evidence_returns_full_metadata():
    """Test storage layer method directly for claim+evidence join."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        storage = JudgeMemoryStorage(JudgeMemoryConfig(sqlite_path=str(db_path)))

        # Create evidence
        evidence = EvidenceRecord(
            evidence_id="ev1",
            content_hash="abc123",
            source_type="expert_report",
            source_title="Expert Analysis",
            source_url="http://expert.com/report",
            content_preview="Expert opinion content.",
            jurisdiction="BC",
            ingested_at=datetime.utcnow(),
        )
        storage.store_evidence(evidence)

        # Create claim linked to evidence
        claim = ClaimRecord(
            claim_id="cl1",
            evidence_id="ev1",
            claim_text="Expert concludes X is true.",
            confidence=0.9,
            status="active",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        storage.store_claim(claim)

        # Search with evidence join (use a word that definitely appears)
        results = storage.search_claims_with_evidence("concludes", limit=10)

        assert len(results) == 1, f"Expected 1 result, got {len(results)}. Results: {results}"
        found_claim, found_evidence, score = results[0]

        # Verify claim data
        assert found_claim.claim_id == "cl1"
        assert found_claim.claim_text == "Expert concludes X is true."

        # Verify evidence metadata is included
        assert found_evidence.evidence_id == "ev1"
        assert found_evidence.source_type == "expert_report"
        assert found_evidence.source_title == "Expert Analysis"
        assert found_evidence.jurisdiction == "BC"
        assert found_evidence.source_url == "http://expert.com/report"

        # Score should be present
        assert 0.0 <= score <= 1.0
