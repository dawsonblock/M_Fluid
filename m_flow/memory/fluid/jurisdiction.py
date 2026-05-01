"""
Fluid Memory Jurisdiction Weighting

Provides per-jurisdiction trust multipliers for the JudgeTracker /
crime-mapping system.

Configuration is loaded from fluid_jurisdictions.yaml (in this directory)
and can be extended by passing custom mappings at runtime.

Usage:
    w = JurisdictionWeighter()
    multiplier = w.weight("US-TX", "court_record")   # e.g. 0.80
    is_auth    = w.is_authoritative("federal", "court_record")  # True
    effective_trust = base_trust * w.weight(jurisdiction, source_type)

The multiplier is applied to source_trust during the effective score
computation so that federal court records rank higher than local blog posts
even when both have the same base source_trust.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

from m_flow.shared.logging_utils import get_logger

logger = get_logger("fluid.jurisdiction")

_YAML_PATH = Path(__file__).parent / "fluid_jurisdictions.yaml"
_AUTHORITATIVE_THRESHOLD = 0.80  # multiplier >= this → authoritative source


def _load_yaml_config(path: Path) -> Dict[str, Dict]:
    """Load jurisdiction YAML config. Returns empty dict on any failure."""
    try:
        import yaml  # type: ignore[import-untyped]
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return {k: v for k, v in data.items() if isinstance(v, dict)}
    except Exception as exc:
        logger.warning("jurisdiction: failed to load %s: %s", path, exc)
        return {}


# Hardcoded fallback (used when YAML fails to load)
_FALLBACK_MULTIPLIERS: Dict[str, float] = {
    "federal":      1.00,
    "supreme_court": 1.00,
    "international": 0.90,
    "state":        0.80,
    "military":     0.85,
    "bankruptcy":   0.75,
    "immigration":  0.75,
    "county":       0.65,
    "local":        0.60,
    "municipal":    0.60,
    "unknown":      0.50,
}

# Source types that bypass jurisdiction weighting (always full trust)
_BYPASS_SOURCE_TYPES = {"court_record", "government_data"}


@lru_cache(maxsize=1)
def _load_multipliers() -> Dict[str, float]:
    """Load and cache jurisdiction multipliers from YAML."""
    config = _load_yaml_config(_YAML_PATH)
    if not config:
        return dict(_FALLBACK_MULTIPLIERS)

    multipliers: Dict[str, float] = {}
    for code, entry in config.items():
        if isinstance(entry, dict) and "trust_multiplier" in entry:
            multipliers[code] = float(entry["trust_multiplier"])

    # Merge fallback for any missing codes
    for code, mult in _FALLBACK_MULTIPLIERS.items():
        multipliers.setdefault(code, mult)

    return multipliers


class JurisdictionWeighter:
    """
    Jurisdiction-aware trust multiplier for the fluid scoring system.

    Jurisdiction multipliers scale source_trust before it enters the
    compute_effective_score formula, so:
        effective_trust = source_trust * jurisdiction_multiplier

    This ensures that a court_record from federal jurisdiction ranks
    higher than the same claim from an unknown local source, even if
    both have the same raw source_trust value.
    """

    def __init__(self, custom_multipliers: Optional[Dict[str, float]] = None) -> None:
        self._base = _load_multipliers()
        self._overrides: Dict[str, float] = custom_multipliers or {}

    def _get_multiplier(self, jurisdiction: str) -> float:
        """Look up multiplier with override > base > unknown fallback."""
        key = (jurisdiction or "unknown").lower().strip()
        if key in self._overrides:
            return self._overrides[key]
        if key in self._base:
            return self._base[key]
        # Try prefix match (e.g. "US-TX-TRAVIS" → "US-TX" → "state")
        parts = key.split("-")
        for i in range(len(parts) - 1, 0, -1):
            prefix = "-".join(parts[:i])
            if prefix in self._base:
                return self._base[prefix]
        return self._base.get("unknown", 0.50)

    def weight(
        self,
        jurisdiction: Optional[str],
        source_type: Optional[str] = None,
    ) -> float:
        """
        Get the trust multiplier for a jurisdiction.

        Args:
            jurisdiction: Jurisdiction code (e.g. "US-TX", "federal", "local")
            source_type: Source type; certain types bypass jurisdiction weighting

        Returns:
            Multiplier in [0.0, 1.0]
        """
        if source_type and source_type.lower() in _BYPASS_SOURCE_TYPES:
            # Court records and government data always get full trust
            return 1.0

        return self._get_multiplier(jurisdiction or "unknown")

    def is_authoritative(
        self,
        jurisdiction: Optional[str],
        source_type: Optional[str] = None,
    ) -> bool:
        """
        Return True if this jurisdiction+source_type combination is authoritative.

        Authoritative means the trust multiplier >= AUTHORITATIVE_THRESHOLD (0.80).

        Args:
            jurisdiction: Jurisdiction code
            source_type: Source type

        Returns:
            True if authoritative, False otherwise
        """
        return self.weight(jurisdiction, source_type) >= _AUTHORITATIVE_THRESHOLD

    def apply(
        self,
        source_trust: float,
        jurisdiction: Optional[str],
        source_type: Optional[str] = None,
    ) -> float:
        """
        Apply jurisdiction weighting to a raw source_trust value.

        Args:
            source_trust: Raw source trust [0, 1]
            jurisdiction: Jurisdiction code
            source_type: Source type

        Returns:
            Weighted trust in [0, 1]
        """
        multiplier = self.weight(jurisdiction, source_type)
        return min(1.0, source_trust * multiplier)

    def list_jurisdictions(self) -> Dict[str, float]:
        """Return all known jurisdiction → multiplier mappings."""
        result = dict(self._base)
        result.update(self._overrides)
        return result


@lru_cache(maxsize=1)
def get_jurisdiction_weighter() -> JurisdictionWeighter:
    """Singleton accessor for the default JurisdictionWeighter."""
    return JurisdictionWeighter()
