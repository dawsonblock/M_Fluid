"""
Judge Memory Fluid Adapter

Thin adapter around fluid memory engine.
Only active if enable_fluid_memory is True.
"""

from typing import Optional, Dict, Any
from datetime import datetime

from m_flow.judge_memory.config import JudgeMemoryConfig


class FluidMemoryAdapter:
    """
    Thin adapter for fluid memory integration.

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

    def _init_engine(self):
        """Initialize fluid memory engine if available."""
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
        except Exception:
            # Fluid memory not available, disable
            self.enabled = False

    def _get_decay_lane(self, source_type: str) -> str:
        """
        Get canonical decay lane for source type.

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
        judge_id: Optional[str] = None,
    ):
        """Touch fluid state for evidence."""
        if not self.enabled or not self._engine:
            return

        try:
            from m_flow.memory.fluid.models import FluidUpdateEvent

            # Get source profile
            source_trust = 0.5
            legal_weight = 0.0
            if self.source_registry:
                profile = await self.source_registry.get_source_profile(source_type)
                source_trust = profile.derive_trust()
                legal_weight = profile.legal_weight

            # Create touch event
            event = FluidUpdateEvent(
                touched_node_ids=[evidence_id],
                source_id=evidence_id,
                source_type=source_type,
                source_trust=source_trust,
                salience=0.5,
                legal_weight=legal_weight,
                decay_lane=self._get_decay_lane(source_type),
                jurisdiction=jurisdiction,
                judge_id=judge_id,
            )

            await self._engine.touch(event)
        except Exception:
            # Fail silently - fluid is optional
            pass

    async def touch_claim(
        self,
        claim_id: str,
        evidence_id: str,
        source_type: str,
        claim_status: str,
    ):
        """Touch fluid state for claim."""
        if not self.enabled or not self._engine:
            return

        try:
            from m_flow.memory.fluid.models import FluidUpdateEvent

            # Get source profile
            source_trust = 0.5
            if self.source_registry:
                profile = await self.source_registry.get_source_profile(source_type)
                source_trust = profile.derive_trust()

            # Create touch event
            event = FluidUpdateEvent(
                touched_node_ids=[claim_id],
                source_id=evidence_id,
                source_type=source_type,
                source_trust=source_trust,
                salience=0.6 if claim_status == "presumed_true" else 0.4,
                decay_lane=self._get_decay_lane(source_type),
            )

            await self._engine.touch(event)
        except Exception:
            # Fail silently - fluid is optional
            pass

    async def get_state(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get fluid state for a node."""
        if not self.enabled or not self._engine:
            return None

        try:
            state = await self._engine.get_state(node_id)
            if state:
                return {
                    "activation": state.activation,
                    "confidence": state.confidence,
                    "source_trust": state.source_trust,
                    "contradiction_pressure": state.contradiction_pressure,
                    "decay_lane": state.decay_lane,
                }
        except Exception:
            pass

        return None
