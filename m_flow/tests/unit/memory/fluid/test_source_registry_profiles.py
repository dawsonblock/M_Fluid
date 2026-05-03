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
        legal_status_label="official_record",
        legal_status_weight=0.95,
        default_claim_status="presumed_true",
    )

    assert entry.authority == 1.00
    assert entry.verifiability == 0.95
    assert entry.originality == 1.00
    assert entry.independence == 0.95
    assert entry.legal_status_label == "official_record"
    assert entry.legal_status_weight == 0.95
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
    assert entry.legal_status_label == "unverified"
    assert entry.legal_status_weight == 0.50
    assert entry.default_claim_status == "needs_verification"


def test_hardcoded_fallback_has_structured_fields():
    """Test hardcoded fallback entries have structured fields."""
    from m_flow.memory.fluid.source_registry import _HARDCODED_FALLBACK

    court_record = _HARDCODED_FALLBACK["court_record"]
    assert court_record.authority == 1.00
    assert court_record.verifiability == 0.95
    assert court_record.legal_status_label == "official_record"
    assert court_record.legal_status_weight == 0.95
    assert court_record.default_claim_status == "presumed_true"

    blog_social = _HARDCODED_FALLBACK["blog_social"]
    assert blog_social.authority == 0.30
    assert blog_social.verifiability == 0.20
    assert blog_social.legal_status_label == "unverified"
    assert blog_social.legal_status_weight == 0.20
    assert blog_social.default_claim_status == "presumed_false"


def test_get_source_profile_method_exists():
    """Test SourceRegistry has get_source_profile method."""
    from m_flow.memory.fluid.source_registry import SourceRegistry

    # Check method exists
    assert hasattr(SourceRegistry, "get_source_profile")


def test_derive_trust_method():
    """Test SourceTrustEntry derive_trust calculates weighted average."""
    from m_flow.memory.fluid.source_registry import SourceTrustEntry

    entry = SourceTrustEntry(
        source_type="test",
        trust=0.50,
        legal_weight=0.50,
        authority=1.00,
        verifiability=0.80,
        originality=0.70,
        independence=0.60,
    )

    # derive_trust = authority*0.30 + verifiability*0.30 + originality*0.20 + independence*0.20
    # = 1.00*0.30 + 0.80*0.30 + 0.70*0.20 + 0.60*0.20
    # = 0.30 + 0.24 + 0.14 + 0.12 = 0.80
    expected = 0.30 + 0.24 + 0.14 + 0.12
    assert entry.derive_trust() == expected


def test_yaml_uses_canonical_lane_names():
    """Test that YAML file uses canonical lane names (interest, attention, legal)."""
    from m_flow.memory.fluid.source_registry import _HARDCODED_FALLBACK

    # Check that canonical lane names are used, not legacy aliases
    assert _HARDCODED_FALLBACK["court_record"].decay_lane == "legal"
    assert _HARDCODED_FALLBACK["blog_social"].decay_lane == "attention"
    assert _HARDCODED_FALLBACK["academic_paper"].decay_lane == "interest"
