"""
Fluid Memory Source Registry

Two-tier source trust registry:
  1. YAML file  — default presets shipped with the package
  2. DB table   — per-deployment overrides (takes priority)

Usage:
    registry = SourceRegistry(store)
    await registry.initialize()
    trust, legal, decay_lane = await registry.get_weights("court_record")
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Dict
from functools import lru_cache

import yaml
from sqlalchemy import String, Float, select, insert, update
from sqlalchemy.orm import Mapped, mapped_column

from m_flow.adapters.relational.ModelBase import Base as ModelBase
from m_flow.memory.fluid.config import get_fluid_config


# ---------------------------------------------------------------------------
# Dataclass for a registry entry
# ---------------------------------------------------------------------------

@dataclass
class SourceTrustEntry:
    source_type: str
    trust: float
    legal_weight: float
    decay_lane: str = "normal"        # "short_term" | "normal" | "legal"
    description: str = ""

    # Structured trust profile fields (v2)
    authority: float = 0.50          # Court/gov authority level
    verifiability: float = 0.50      # Can be independently verified
    originality: float = 0.50        # Primary vs secondary source
    independence: float = 0.50      # Independent vs affiliated
    legal_status: str = "unverified"  # official_record, hearsay, etc.
    default_claim_status: str = "needs_verification"  # presumed_true, needs_verification, presumed_false


# ---------------------------------------------------------------------------
# SQLAlchemy model
# ---------------------------------------------------------------------------

class SourceRegistryRecord(ModelBase):
    """DB table for per-deployment source trust overrides."""

    __tablename__ = "fluid_source_registry"

    source_type: Mapped[str] = mapped_column(String(100), primary_key=True)
    trust: Mapped[float] = mapped_column(Float, default=0.10)
    legal_weight: Mapped[float] = mapped_column(Float, default=0.00)
    decay_lane: Mapped[str] = mapped_column(String(20), default="normal")
    description: Mapped[str] = mapped_column(String(500), default="")


# ---------------------------------------------------------------------------
# Default YAML path
# ---------------------------------------------------------------------------

_DEFAULT_YAML = os.path.join(
    os.path.dirname(__file__), "fluid_sources.yaml"
)

_HARDCODED_FALLBACK: Dict[str, SourceTrustEntry] = {
    "court_record":      SourceTrustEntry("court_record",      0.95, 1.00, "legal",      "Court judgments, orders, official records",      1.00, 0.95, 1.00, 0.95, "official_record", "presumed_true"),
    "government_data":   SourceTrustEntry("government_data",   0.85, 0.80, "legal",      "Government datasets and publications",           0.95, 0.85, 0.90, 0.80, "official_record", "presumed_true"),
    "police_release":    SourceTrustEntry("police_release",    0.80, 0.70, "legal",      "Official police press releases",                  0.85, 0.75, 0.95, 0.70, "official_record", "needs_verification"),
    "academic_paper":    SourceTrustEntry("academic_paper",    0.75, 0.40, "normal",     "Peer-reviewed academic publications",              0.80, 0.85, 0.90, 0.75, "verified", "needs_verification"),
    "expert_report":     SourceTrustEntry("expert_report",     0.70, 0.50, "normal",     "Expert witness reports and assessments",          0.75, 0.70, 0.85, 0.70, "verified", "needs_verification"),
    "witness_statement": SourceTrustEntry("witness_statement", 0.50, 0.40, "normal",     "Witness statements and depositions",                0.60, 0.50, 0.95, 0.85, "hearsay", "needs_verification"),
    "mainstream_news":   SourceTrustEntry("mainstream_news",   0.60, 0.30, "normal",     "Established news organisations",                   0.65, 0.60, 0.70, 0.60, "unverified", "needs_verification"),
    "blog_social":       SourceTrustEntry("blog_social",       0.25, 0.05, "short_term", "Blogs, social media, unverified sources",          0.30, 0.20, 0.60, 0.40, "unverified", "presumed_false"),
    "unknown":           SourceTrustEntry("unknown",           0.10, 0.00, "normal",     "Unclassified or unknown source type",             0.50, 0.50, 0.50, 0.50, "unverified", "needs_verification"),
}


# ---------------------------------------------------------------------------
# Registry class
# ---------------------------------------------------------------------------

class SourceRegistry:
    """
    Two-tier source trust registry.

    Resolution order for any lookup:
        1. DB table  (fluid_source_registry)
        2. YAML file (<module_dir>/fluid_sources.yaml)
        3. Hardcoded fallback dict
    """

    def __init__(self, engine) -> None:
        self._engine = engine
        self._yaml_cache: Dict[str, SourceTrustEntry] = {}
        self._db_cache: Dict[str, SourceTrustEntry] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Load YAML and warm DB cache."""
        if self._initialized:
            return
        self._yaml_cache = _load_yaml_registry()
        self._db_cache = await self._load_db_registry()
        self._initialized = True

    async def get_weights(self, source_type: Optional[str]) -> tuple[float, float, str]:
        """
        Return (trust, legal_weight, decay_lane) for a source type.

        Resolves: DB → YAML → hardcoded fallback.
        """
        if not self._initialized:
            await self.initialize()

        key = (source_type or "unknown").lower().replace(" ", "_")

        entry = (
            self._db_cache.get(key)
            or self._yaml_cache.get(key)
            or _HARDCODED_FALLBACK.get(key)
            or _HARDCODED_FALLBACK["unknown"]
        )
        return entry.trust, entry.legal_weight, entry.decay_lane

    async def get_source_profile(self, source_type: Optional[str]) -> SourceTrustEntry:
        """
        Return full structured trust profile for a source type.

        Resolves: DB → YAML → hardcoded fallback.
        """
        if not self._initialized:
            await self.initialize()

        key = (source_type or "unknown").lower().replace(" ", "_")

        return (
            self._db_cache.get(key)
            or self._yaml_cache.get(key)
            or _HARDCODED_FALLBACK.get(key)
            or _HARDCODED_FALLBACK["unknown"]
        )

    async def upsert(self, entry: SourceTrustEntry) -> None:
        """Insert or update a source entry in the DB (runtime override)."""
        async with self._engine.sessionmaker() as session:
            result = await session.execute(
                update(SourceRegistryRecord)
                .where(SourceRegistryRecord.source_type == entry.source_type)
                .values(
                    trust=entry.trust,
                    legal_weight=entry.legal_weight,
                    decay_lane=entry.decay_lane,
                    description=entry.description,
                )
            )
            if result.rowcount == 0:
                await session.execute(
                    insert(SourceRegistryRecord).values(
                        source_type=entry.source_type,
                        trust=entry.trust,
                        legal_weight=entry.legal_weight,
                        decay_lane=entry.decay_lane,
                        description=entry.description,
                    )
                )
            await session.commit()
        # Invalidate cache
        self._db_cache[entry.source_type] = entry

    async def _load_db_registry(self) -> Dict[str, SourceTrustEntry]:
        """Load all source entries from the DB."""
        try:
            async with self._engine.sessionmaker() as session:
                result = await session.execute(select(SourceRegistryRecord))
                records = result.scalars().all()
                return {
                    r.source_type: SourceTrustEntry(
                        source_type=r.source_type,
                        trust=r.trust,
                        legal_weight=r.legal_weight,
                        decay_lane=r.decay_lane,
                        description=r.description,
                    )
                    for r in records
                }
        except Exception:
            return {}


def _load_yaml_registry() -> Dict[str, SourceTrustEntry]:
    """Load source trust presets from YAML file."""
    yaml_path = _DEFAULT_YAML
    if not os.path.exists(yaml_path):
        return {}

    try:
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f) or {}

        entries: Dict[str, SourceTrustEntry] = {}
        for source_type, cfg in data.get("sources", {}).items():
            entries[source_type] = SourceTrustEntry(
                source_type=source_type,
                trust=float(cfg.get("trust", 0.10)),
                legal_weight=float(cfg.get("legal_weight", 0.00)),
                decay_lane=str(cfg.get("decay_lane", "normal")),
                description=str(cfg.get("description", "")),
                authority=float(cfg.get("authority", 0.50)),
                verifiability=float(cfg.get("verifiability", 0.50)),
                originality=float(cfg.get("originality", 0.50)),
                independence=float(cfg.get("independence", 0.50)),
                legal_status=str(cfg.get("legal_status", "unverified")),
                default_claim_status=str(cfg.get("default_claim_status", "needs_verification")),
            )
        return entries
    except Exception:
        return {}
