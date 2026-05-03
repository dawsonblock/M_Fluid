"""Judge Memory Fluid Adapter

Thin adapter around the standalone fluid_memory engine.
Never imports from m_flow — uses only the fluid_memory package in this repo.
Only active when JudgeMemoryConfig.enable_fluid_memory is True.
"""

import hashlib
from typing import Optional, Dict, Any

from judge_memory.config import JudgeMemoryConfig
from judge_memory._logger import get_logger

logger = get_logger(__name__)

# Try to import standalone fluid_memory (no m_flow required)
HAS_FLUID = False
try:
    from fluid_memory import FluidMemoryEngine, FluidMemoryConfig

    HAS_FLUID = True
    logger.debug("Standalone fluid_memory available")
except ImportError as e:
    logger.debug(f"fluid_memory not importable ({e}); fluid adapter will stay disabled")

# Canonical source trust profiles — single source of truth for this module.
HARDCODED_SOURCE_PROFILES: Dict[str, Any] = {
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
        "legal_status_weight": 0.6,
        "default_claim_status": "needs_verification",
    },
    "expert_report": {
        "authority": 0.8,
        "verifiability": 0.8,
        "originality": 0.75,
        "independence": 0.65,
        "legal_status_label": "expert_testimony",
        "legal_status_weight": 0.7,
        "default_claim_status": "needs_verification",
    },
    "witness_statement": {
        "authority": 0.6,
        "verifiability": 0.6,
        "originality": 0.9,
        "independence": 0.5,
        "legal_status_label": "eyewitness_account",
        "legal_status_weight": 0.5,
        "default_claim_status": "needs_verification",
    },
    "mainstream_news": {
        "authority": 0.6,
        "verifiability": 0.7,
        "originality": 0.5,
        "independence": 0.6,
        "legal_status_label": "press_report",
        "legal_status_weight": 0.4,
        "default_claim_status": "needs_verification",
    },
    "blog_social": {
        "authority": 0.3,
        "verifiability": 0.3,
        "originality": 0.6,
        "independence": 0.4,
        "legal_status_label": "unverified_source",
        "legal_status_weight": 0.2,
        "default_claim_status": "unverified",
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


class FluidMemoryAdapter:
    """Thin adapter for fluid memory integration.

    Required behavior:
    - If enable_fluid_memory=False, do nothing
    - If enabled, update/touch fluid state for evidence and claims
    - Use SourceRegistry for source profile
    - Use canonical decay lanes
    - Never let media amplification increase trust
    - Contradiction pressure must not delete or hide evidence
    """

    def __init__(self, config: JudgeMemoryConfig, source_registry=None):
        self.config = config
        self.enabled = config.enable_fluid_memory
        self.source_registry = source_registry
        self._engine = None

        if self.enabled:
            self._init_engine()

    def _init_engine(self) -> None:
        """Initialize standalone fluid memory engine."""
        if not HAS_FLUID:
            logger.warning("fluid_memory not importable; fluid adapter disabled")
            self.enabled = False
            return

        try:
            cfg = FluidMemoryConfig()
            self._engine = FluidMemoryEngine(cfg, enable_audit=False)
            logger.info("Standalone fluid memory engine initialized")
        except Exception as e:
            logger.warning(f"Fluid memory initialization failed: {e}")
            self.enabled = False

    def _get_decay_lane(self, source_type: str) -> str:
        """Get canonical decay lane for source type.

        Canonical lanes:
        - court_record -> legal
        - government_data -> legal or trust
        - police_release -> trust
        - mainstream_news -> interest
        - blog_social -> attention
        - unknown -> attention
        """
        lane_mapping = {
            "court_record": "legal",
            "government_data": "legal",
            "police_release": "trust",
            "academic_paper": "interest",
            "expert_report": "interest",
            "witness_statement": "interest",
            "mainstream_news": "interest",
            "blog_social": "attention",
            "unknown": "attention",
        }
        return lane_mapping.get(source_type, "interest")

    def _touch_fluid_node(
        self,
        node_id: str,
        tags: list,
        salience: float,
        confidence: float,
        metadata: Dict[str, Any],
    ) -> None:
        """Create or reinforce the fluid node for a judge node_id.

        First touch creates the memory item with calibrated salience/confidence.
        Subsequent touches reinforce the existing item.
        All exceptions are suppressed — fluid state is advisory, not required.
        """
        content_hash = hashlib.sha256(node_id.encode("utf-8")).hexdigest()
        existing = self._engine.storage.get_memory_by_hash(content_hash)
        if existing:
            self._engine.reinforce(existing.memory_id, metadata=metadata)
            return

        from fluid_memory.exceptions import DuplicateMemoryError

        try:
            self._engine.add_memory(
                content=node_id,
                tags=tags,
                salience=salience,
                confidence=confidence,
                metadata=metadata,
                detect_contradictions=False,
            )
        except DuplicateMemoryError:
            # Race: another thread created the node between our check and add
            existing = self._engine.storage.get_memory_by_hash(content_hash)
            if existing:
                self._engine.reinforce(existing.memory_id, metadata=metadata)

    async def touch_evidence(
        self,
        evidence_id: str,
        source_type: str,
        jurisdiction: Optional[str] = None,
        judge_id: Optional[str] = None,
    ) -> None:
        """Touch fluid state for an evidence record."""
        if not self.enabled or not self._engine:
            return
        try:
            profile = self.get_source_profile(source_type)
            salience = (
                profile["authority"] * 0.30
                + profile["verifiability"] * 0.30
                + profile["originality"] * 0.20
                + profile["independence"] * 0.20
            )
            confidence = profile.get("legal_status_weight", 0.5)
            meta: Dict[str, Any] = {
                "source_type": source_type,
                "decay_lane": self._get_decay_lane(source_type),
            }
            if jurisdiction:
                meta["jurisdiction"] = jurisdiction
            if judge_id:
                meta["judge_id"] = judge_id
            self._touch_fluid_node(
                evidence_id,
                tags=[source_type, "evidence"],
                salience=salience,
                confidence=confidence,
                metadata=meta,
            )
            logger.debug(f"Fluid state touched for evidence {evidence_id}")
        except Exception as e:
            logger.warning(
                f"Failed to touch fluid state for evidence {evidence_id}: {e}"
            )

    async def touch_claim(
        self,
        claim_id: str,
        evidence_id: str,
        source_type: str,
        claim_status: str,
        jurisdiction: Optional[str] = None,
        judge_id: Optional[str] = None,
    ) -> None:
        """Touch fluid state for a claim."""
        if not self.enabled or not self._engine:
            return
        try:
            profile = self.get_source_profile(source_type)
            confidence = profile.get("legal_status_weight", 0.5)
            claim_salience = (
                0.6 if claim_status in {"confirmed", "presumed_true"} else 0.4
            )
            meta: Dict[str, Any] = {
                "source_type": source_type,
                "decay_lane": self._get_decay_lane(source_type),
                "evidence_id": evidence_id,
            }
            if jurisdiction:
                meta["jurisdiction"] = jurisdiction
            if judge_id:
                meta["judge_id"] = judge_id
            self._touch_fluid_node(
                claim_id,
                tags=[source_type, "claim"],
                salience=claim_salience,
                confidence=confidence,
                metadata=meta,
            )
            logger.debug(f"Fluid state touched for claim {claim_id}")
        except Exception as e:
            logger.warning(f"Failed to touch fluid state for claim {claim_id}: {e}")

    async def get_state(self, memory_id: str):
        """Return fluid state for a node when fluid memory is enabled.

        In disabled or unavailable mode this intentionally returns None so
        judge_memory remains isolated from m_flow.
        """
        if not self.enabled or not self._engine:
            return None

        getter = getattr(self._engine, "get_state", None)
        if getter is None:
            return None

        result = getter(memory_id)
        if hasattr(result, "__await__"):
            return await result
        return result

    def get_source_profile(self, source_type: str) -> Dict[str, Any]:
        """Get source profile for trust calculations.

        Uses hardcoded profiles to maintain zero external dependencies.
        The source_registry parameter passed to __init__ is reserved for
        future m_flow integration but currently unused to ensure
        judge_memory remains standalone and functional without the
        full M-flow runtime.

        Args:
            source_type: Type of source (court_record, government_data, etc.)

        Returns:
            Source profile dict with authority, verifiability, originality,
            independence, legal_status_label, legal_status_weight, and
            default_claim_status.
        """
        if self.source_registry is not None:
            try:
                return self.source_registry.get_profile(source_type)
            except Exception:
                pass
        return HARDCODED_SOURCE_PROFILES.get(
            source_type, HARDCODED_SOURCE_PROFILES["unknown"]
        )
