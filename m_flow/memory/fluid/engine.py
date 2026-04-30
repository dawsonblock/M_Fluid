"""
Fluid Memory Engine

Main orchestrator for the fluid memory system.
Runs after episodic writes and before retrieval.

All behaviour is governed by FluidMemoryConfig (env vars).
Failures are logged and never propagate to callers.
"""

from time import time
from typing import TYPE_CHECKING, Optional, List

from m_flow.shared.logging_utils import get_logger
from m_flow.memory.fluid.models import FluidUpdateEvent, FluidMemoryState
from m_flow.memory.fluid.config import get_fluid_config
from m_flow.memory.fluid.state_store import FluidStateStore
from m_flow.memory.fluid.propagation import propagate_activation_episodic
from m_flow.memory.fluid.decay import apply_decay, update_recency_scores, _LANE_RATES, NORMAL_DECAY
from m_flow.memory.fluid.contradiction import apply_contradictions
from m_flow.memory.fluid.audit import FluidAuditLogger, AuditEventType, FluidProvenance

if TYPE_CHECKING:
    from m_flow.adapters.graph.graph_db_interface import GraphProvider

logger = get_logger("fluid.engine")


class FluidMemoryEngine:
    """
    Fluid Memory Engine — orchestrates fluid state updates.

    The engine receives touch events from episodic memory writes,
    updates fluid state, applies decay and contradictions,
    and propagates activation through the graph.

    All settings are read from FluidMemoryConfig at construction time
    so environment variable overrides take effect without restarting.

    Usage:
        engine = FluidMemoryEngine(graph_engine, store)
        await engine.touch(FluidUpdateEvent(
            touched_node_ids=[episode.id, facet.id, entity.id],
            source_id=document_id,
            source_type="mainstream_news",
            source_trust=0.60,
            salience=0.7,
            legal_weight=0.3,
            decay_lane="normal",
        ))
    """

    def __init__(
        self,
        graph_engine: "GraphProvider",
        store: FluidStateStore,
        enable_audit: Optional[bool] = None,
    ):
        cfg = get_fluid_config()
        self.graph = graph_engine
        self.store = store

        _audit_on = enable_audit if enable_audit is not None else cfg.enable_audit
        self.audit = FluidAuditLogger(store) if _audit_on else None

        self._activation_increment = cfg.activation_increment
        self._max_activation = cfg.max_activation
        self._propagation_start = cfg.propagation_start_activation
        self._propagation_depth = cfg.propagation_max_depth
        self._min_activation = cfg.minimum_activation
        self._enable_contradiction = cfg.enable_contradiction

    async def touch(self, event: FluidUpdateEvent) -> None:
        """
        Process a touch event — the main entry point.

        Called after episodic memory is written.  Steps:
        1. Load current states for all touched node IDs
        2. Boost activation, update trust/salience/legal_weight, set decay lane
        3. Apply decay (per-day, lane-aware)
        4. Apply legacy list-based contradiction pressure
        5. Run LLM-assisted contradiction detection (if enabled)
        6. Persist updated states + any new ClaimConflicts
        7. Audit log all mutations with provenance
        8. Propagate activation ripple through graph (BFS, depth 2)

        Args:
            event: The fluid update event carrying source metadata
        """
        now = time()
        cfg = get_fluid_config()

        # Step 1: Load current states
        states = await self.store.get_many(event.touched_node_ids)

        # Snapshot activations before mutation for provenance
        old_activations = {s.node_id: s.activation for s in states}

        # Step 2: Boost and update metadata
        decay_lane = event.decay_lane or "normal"
        decay_rate = _LANE_RATES.get(decay_lane, NORMAL_DECAY)

        for state in states:
            state.activation = min(
                self._max_activation,
                state.activation + self._activation_increment,
            )
            state.reinforcement_count += 1
            state.source_trust = max(state.source_trust, event.source_trust)
            state.salience = max(state.salience, event.salience)
            state.legal_weight = max(state.legal_weight, event.legal_weight)
            state.last_touched_at = now
            state.recency_score = 1.0
            state.decay_lane = decay_lane
            state.decay_rate = decay_rate

        # Step 3: Apply per-day decay
        states = apply_decay(states, now, min_activation=self._min_activation)

        # Step 4: Legacy list-based contradiction pressure
        states = apply_contradictions(states, event)

        # Step 5: LLM-assisted contradiction detection
        if self._enable_contradiction and cfg.enable_contradiction:
            await self._run_contradiction_detection(event, states, now)

        # Step 6: Persist updated states
        await self.store.upsert_many(states)

        # Step 7: Audit log with provenance
        if self.audit:
            for state in states:
                old_act = old_activations.get(state.node_id)
                prov = FluidProvenance(
                    node_id=state.node_id,
                    event_type=AuditEventType.TOUCH,
                    source_document_id=event.source_id,
                    source_type=event.source_type,
                    extractor="episodic_write",
                    confidence_method="source_trust",
                    old_values={"activation": old_act} if old_act is not None else {},
                    new_values={"activation": state.activation},
                )
                await self.audit.log_touch(
                    node_id=state.node_id,
                    new_activation=state.activation,
                    source_id=event.source_id,
                    source_type=event.source_type,
                    old_activation=old_act,
                    provenance=prov,
                )

        # Step 8: Propagate activation ripple through graph
        try:
            ripple_updates = await propagate_activation_episodic(
                graph_engine=self.graph,
                seed_node_ids=event.touched_node_ids,
                start_activation=self._propagation_start,
                max_depth=self._propagation_depth,
            )
        except Exception as exc:
            logger.warning("fluid.engine: propagation failed: %s", exc)
            ripple_updates = {}

        if ripple_updates:
            merge_results = await self.store.merge_activation(
                ripple_updates,
                merge_mode="max",
            )
            if self.audit:
                for result in merge_results:
                    if result.merged and result.new_activation > result.old_activation:
                        await self.audit.log_propagation(
                            node_id=result.node_id,
                            old_activation=result.old_activation,
                            new_activation=result.new_activation,
                            source_nodes=event.touched_node_ids,
                        )

    async def _run_contradiction_detection(
        self,
        event: FluidUpdateEvent,
        states: List[FluidMemoryState],
        now: float,
    ) -> None:
        """
        Run LLM-assisted contradiction detection for each touched node.

        Fetches existing graph neighbours, runs the ContradictionDetector,
        persists ClaimConflict records, and increases contradiction_pressure
        on both conflicting nodes.
        """
        try:
            from m_flow.memory.fluid.contradiction_detector import ContradictionDetector

            detector = ContradictionDetector(self.graph)
            state_map = {s.node_id: s for s in states}

            for state in states:
                node_text = await self._fetch_node_text(state.node_id)
                if not node_text:
                    continue

                # Gather candidate node IDs from graph neighbourhood
                candidates = await self._get_neighbour_ids(state.node_id, limit=5)
                if not candidates:
                    continue

                conflicts = await detector.detect(
                    new_node_id=state.node_id,
                    new_node_text=node_text,
                    new_source_id=event.source_id,
                    new_source_type=event.source_type,
                    candidate_node_ids=candidates,
                )

                for conflict in conflicts:
                    await self.store.save_claim_conflict(conflict)

                    # Apply contradiction pressure to both nodes
                    for nid in (conflict.node_id_a, conflict.node_id_b):
                        s = state_map.get(nid)
                        if s:
                            old_pressure = s.contradiction_pressure
                            s.contradiction_pressure = min(
                                1.0,
                                s.contradiction_pressure + 0.15 * conflict.confidence,
                            )
                            if self.audit:
                                await self.audit.log_contradiction(
                                    node_id=nid,
                                    new_pressure=s.contradiction_pressure,
                                    contradicting_source=event.source_id or "unknown",
                                    old_pressure=old_pressure,
                                    conflict_reason=conflict.conflict_reason,
                                )

        except Exception as exc:
            logger.warning("fluid.engine: contradiction detection failed: %s", exc)

    async def _fetch_node_text(self, node_id: str) -> Optional[str]:
        """Fetch a node's summary text from the graph."""
        try:
            rows = await self.graph.query(
                "MATCH (n {id: $id}) "
                "RETURN coalesce(n.summary, n.search_text, n.name, '') AS text LIMIT 1",
                {"id": node_id},
            )
            if rows:
                return str(rows[0].get("text", "")).strip() or None
        except Exception:
            pass
        return None

    async def _get_neighbour_ids(self, node_id: str, limit: int = 5) -> List[str]:
        """Get IDs of nodes connected to node_id via any edge."""
        try:
            rows = await self.graph.query(
                "MATCH (n {id: $id})--(m) RETURN m.id AS nid LIMIT $limit",
                {"id": node_id, "limit": limit},
            )
            return [str(r["nid"]) for r in rows if r.get("nid")]
        except Exception:
            return []

    async def get_state(self, node_id: str) -> Optional[FluidMemoryState]:
        """Get fluid state for a single node."""
        return await self.store.get(node_id)

    async def get_states(self, node_ids: List[str]) -> List[FluidMemoryState]:
        """Get fluid states for multiple nodes."""
        return await self.store.get_many(node_ids)

    async def manual_activate(
        self,
        node_id: str,
        activation_amount: float = 0.5,
        reason: str = "manual",
    ) -> None:
        """
        Manually activate a node (admin/debug use).

        Args:
            node_id: Node to activate
            activation_amount: Amount to add to activation
            reason: Reason for manual activation (recorded in audit)
        """
        state = await self.store.get(node_id)
        if not state:
            state = FluidMemoryState(node_id=node_id)

        old_activation = state.activation
        state.activation = min(self._max_activation, state.activation + activation_amount)
        state.last_touched_at = time()

        await self.store.upsert(state)

        if self.audit:
            prov = FluidProvenance(
                node_id=node_id,
                event_type=AuditEventType.MANUAL,
                extractor="manual",
                confidence_method="manual",
                old_values={"activation": old_activation},
                new_values={"activation": state.activation},
            )
            await self.audit.log_event(
                event_type=AuditEventType.MANUAL,
                node_id=node_id,
                field_changed="activation",
                old_value=old_activation,
                new_value=state.activation,
                provenance=prov,
                metadata={"reason": reason},
            )

    async def decay_all(self, threshold: float = 0.05) -> int:
        """
        Apply per-day decay to all active nodes above threshold.

        Typically called by a scheduled background job (hourly/daily).
        Only activation and recency_score are modified.

        Args:
            threshold: Minimum activation to consider for decay

        Returns:
            Number of nodes processed
        """
        states = await self.store.get_all_with_activation_above(threshold)

        if not states:
            return 0

        now = time()
        old_activations = {s.node_id: s.activation for s in states}

        decayed_states = apply_decay(states, now, min_activation=self._min_activation)
        decayed_states = update_recency_scores(decayed_states, now)

        await self.store.upsert_many(decayed_states)

        if self.audit:
            for state in decayed_states:
                await self.audit.log_decay(
                    node_id=state.node_id,
                    new_activation=state.activation,
                    old_activation=old_activations.get(state.node_id),
                )

        return len(decayed_states)

    async def get_active_nodes(
        self,
        min_activation: float = 0.1,
    ) -> List[FluidMemoryState]:
        """
        Get all currently active nodes above min_activation.

        Args:
            min_activation: Minimum activation threshold

        Returns:
            List of active fluid states, sorted by activation descending
        """
        return await self.store.get_all_with_activation_above(min_activation)
