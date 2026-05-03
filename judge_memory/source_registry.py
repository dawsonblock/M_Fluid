"""Judge Memory Source Registry

Single source of truth for source type trust profiles.
Replaces the scattered SOURCE_AUTHORITY dicts across search.py and fluid_adapter.py.
"""

from typing import Optional, Dict, Any

# Full trust profiles — one canonical dict for the whole judge_memory package.
_DEFAULT_PROFILES: Dict[str, Dict[str, Any]] = {
    "court_record": {
        "authority": 0.9,
        "verifiability": 0.95,
        "originality": 0.9,
        "independence": 0.85,
        "legal_status_label": "primary_authority",
        "legal_status_weight": 0.95,
        "default_claim_status": "presumed_valid",
    },
    "government_data": {
        "authority": 0.85,
        "verifiability": 0.9,
        "originality": 0.8,
        "independence": 0.7,
        "legal_status_label": "official_record",
        "legal_status_weight": 0.85,
        "default_claim_status": "presumed_valid",
    },
    "police_release": {
        "authority": 0.7,
        "verifiability": 0.75,
        "originality": 0.7,
        "independence": 0.6,
        "legal_status_label": "official_statement",
        "legal_status_weight": 0.7,
        "default_claim_status": "needs_verification",
    },
    "academic_paper": {
        "authority": 0.75,
        "verifiability": 0.85,
        "originality": 0.8,
        "independence": 0.75,
        "legal_status_label": "expert_opinion",
        "legal_status_weight": 0.75,
        "default_claim_status": "needs_verification",
    },
    "expert_report": {
        "authority": 0.8,
        "verifiability": 0.8,
        "originality": 0.75,
        "independence": 0.7,
        "legal_status_label": "expert_opinion",
        "legal_status_weight": 0.8,
        "default_claim_status": "needs_verification",
    },
    "witness_statement": {
        "authority": 0.6,
        "verifiability": 0.65,
        "originality": 0.7,
        "independence": 0.6,
        "legal_status_label": "testimonial",
        "legal_status_weight": 0.6,
        "default_claim_status": "needs_verification",
    },
    "mainstream_news": {
        "authority": 0.5,
        "verifiability": 0.6,
        "originality": 0.5,
        "independence": 0.5,
        "legal_status_label": "media_report",
        "legal_status_weight": 0.45,
        "default_claim_status": "needs_verification",
    },
    "blog_social": {
        "authority": 0.2,
        "verifiability": 0.25,
        "originality": 0.3,
        "independence": 0.2,
        "legal_status_label": "unverified_media",
        "legal_status_weight": 0.15,
        "default_claim_status": "speculative",
    },
    "unknown": {
        "authority": 0.5,
        "verifiability": 0.5,
        "originality": 0.5,
        "independence": 0.5,
        "legal_status_label": "unverified",
        "legal_status_weight": 0.5,
        "default_claim_status": "needs_verification",
    },
}

_FALLBACK = _DEFAULT_PROFILES["unknown"]


class SourceRegistry:
    """Registry of source type trust profiles.

    Provides a single source of truth for authority weights and full trust
    profiles used across search ranking and fluid memory calibration.

    Args:
        profiles: Optional override dict of source_type -> profile dicts.
                  Merged on top of the defaults — unknown keys fall back to
                  the built-in ``_DEFAULT_PROFILES``.
    """

    def __init__(self, profiles: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        if profiles:
            self._profiles = {**_DEFAULT_PROFILES, **profiles}
        else:
            self._profiles = _DEFAULT_PROFILES

    def get_profile(self, source_type: Optional[str]) -> Dict[str, Any]:
        """Return the full trust profile for *source_type*.

        Never raises — unknown types return the ``unknown`` fallback profile.
        """
        return self._profiles.get(source_type or "unknown", _FALLBACK)

    def get_authority(self, source_type: Optional[str]) -> float:
        """Return the authority weight (0–1) for *source_type*."""
        return self.get_profile(source_type).get("authority", 0.5)

    def known_types(self) -> list:
        """Return sorted list of registered source type names."""
        return sorted(self._profiles.keys())


# Package-level default instance — import and use directly when no custom
# profiles are needed.
DEFAULT_REGISTRY: SourceRegistry = SourceRegistry()
