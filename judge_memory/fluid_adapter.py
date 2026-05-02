"""Judge Memory Fluid Adapter

Thin adapter around fluid memory engine.
Only active if enable_fluid_memory is True.
Uses defensive imports to avoid hard dependency on m_flow.
"""

from typing import Optional, Dict, Any
from datetime import datetime

from judge_memory.config import JudgeMemoryConfig
from judge_memory._logger import get_logger

logger = get_logger(__name__)

# Defensive imports - fluid memory is optional
HAS_FLUID = False
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

# Try to import m_flow fluid memory (optional)
# Catch all exceptions because m_flow may have internal errors
try:
    from m_flow.memory.fluid.source_registry import (
        _HARDCODED_FALLBACK as FLUID_SOURCE_PROFILES,
    )

    HAS_FLUID = True
    logger.debug("Fluid memory integration available")
except Exception as e:
    logger.debug(f"Fluid memory not available ({type(e).__name__}), using local profiles")
    FLUID_SOURCE_PROFILES = HARDCODED_SOURCE_PROFILES


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
        """Initialize fluid memory engine if available."""
        if not HAS_FLUID:
            logger.warning("Fluid memory requested but m_flow not available")
            self.enabled = False
            return

        try:
            from m_flow.memory.fluid.engine import FluidMemoryEngine
            from m_flow.memory.fluid.state_store import FluidStateStore
            from m_flow.memory.fluid.config import get_fluid_config

            fluid_cfg = get_fluid_config()
            if not fluid_cfg.enable:
                self.enabled = False
                return

            # Create store
            store = FluidStateStore()

            # Create engine (without graph for now)
            self._engine = FluidMemoryEngine(graph_engine=None, store=store)
            logger.info("Fluid memory engine initialized")
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

    async def touch_evidence(
        self,
        evidence_id: str,
        source_type: str,
        jurisdiction: Optional[str] = None,
    ) -> None:
        """Touch fluid state for evidence.
        
        Args:
            evidence_id: Evidence record ID
            source_type: Source type for decay lane selection
            jurisdiction: Optional jurisdiction
        """
        if not self.enabled:
            return

        lane = self._get_decay_lane(source_type)
        logger.debug(f"Touching evidence {evidence_id} in lane {lane}")

        # TODO: Implement actual fluid state touch when engine available
        # This would call self._engine.touch() with appropriate parameters

    def get_source_profile(self, source_type: str) -> Dict[str, Any]:
        """Get source profile for trust calculations.
        
        Args:
            source_type: Type of source
            
        Returns:
            Source profile dict with authority, verifiability, etc.
        """
        # Use hardcoded profiles (no dependency on m_flow)
        return HARDCODED_SOURCE_PROFILES.get(
            source_type, HARDCODED_SOURCE_PROFILES["unknown"]
        )
