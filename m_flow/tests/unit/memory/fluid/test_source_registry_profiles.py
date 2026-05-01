"""
Test SourceRegistry structured trust profiles.
Tests that SourceTrustEntry has structured fields and get_source_profile works.
"""

import pytest
from dataclasses import dataclass


def test_source_trust_entry_structured_fields():
    """Test SourceTrustEntry has structured trust profile fields."""
    from m_flow.memory.fluid.source_registry import SourceTrustEntry

    entry = SourceTrustEntry(
        source_type="court_record",
        trust=0.95,
        legal_weight=1.00,
        decay_lane="legal",
        description="Court judgments",
        authority=1.00,
        verifiability=0.95,
        originality=1.00,
        independence=0.95,
        legal_status="official_record",
        default_claim_status="presumed_true",
    )

    assert entry.authority == 1.00
    assert entry.verifiability == 0.95
    assert entry.originality == 1.00
    assert entry.independence == 0.95
    assert entry.legal_status == "official_record"
    assert entry.default_claim_status == "presumed_true"


def test_source_trust_entry_defaults():
    """Test SourceTrustEntry structured fields have defaults."""
    from m_flow.memory.fluid.source_registry import SourceTrustEntry

    entry = SourceTrustEntry(
        source_type="unknown",
        trust=0.10,
        legal_weight=0.00,
    )

    # Check defaults
    assert entry.authority == 0.50
    assert entry.verifiability == 0.50
    assert entry.originality == 0.50
    assert entry.independence == 0.50
    assert entry.legal_status == "unverified"
    assert entry.default_claim_status == "needs_verification"


def test_hardcoded_fallback_has_structured_fields():
    """Test hardcoded fallback entries have structured fields."""
    from m_flow.memory.fluid.source_registry import _HARDCODED_FALLBACK

    court_record = _HARDCODED_FALLBACK["court_record"]
    assert court_record.authority == 1.00
    assert court_record.verifiability == 0.95
    assert court_record.legal_status == "official_record"
    assert court_record.default_claim_status == "presumed_true"

    blog_social = _HARDCODED_FALLBACK["blog_social"]
    assert blog_social.authority == 0.30
    assert blog_social.verifiability == 0.20
    assert blog_social.legal_status == "unverified"
    assert blog_social.default_claim_status == "presumed_false"


def test_get_source_profile_method_exists():
    """Test SourceRegistry has get_source_profile method."""
    from m_flow.memory.fluid.source_registry import SourceRegistry

    # Check method exists
    assert hasattr(SourceRegistry, "get_source_profile")
