"""
Fluid Memory Service Interface

Defines the Protocol that both the local (in-process) implementation
and a future remote (HTTP/gRPC) implementation must satisfy.

Both integration points — write_episodic_memories.py and bundle_search.py —
call through this interface so that the fluid layer can be extracted into
a standalone service later without changing the call sites.

Usage (local, default):
    service = LocalFluidMemoryService(graph_engine, store)
    await service.touch(event)
    bundles = await service.apply_fluid_scores(bundles)

Future (remote):
    service = RemoteFluidMemoryService(base_url="http://fluid-memory-svc:8080")
    await service.touch(event)
    bundles = await service.apply_fluid_scores(bundles)
"""

from __future__ import annotations

from typing import Optional, List, Protocol, runtime_checkable, TYPE_CHECKING

from m_flow.memory.fluid.models import FluidUpdateEvent, FluidMemoryState

if TYPE_CHECKING:
    from m_flow.memory.fluid.state_store import FluidStateStore
    from m_flow.adapters.graph.graph_db_interface import GraphProvider


# ---------------------------------------------------------------------------
# Protocol (extraction boundary)
# ---------------------------------------------------------------------------

@runtime_checkable
class FluidMemoryServiceInterface(Protocol):
    """
    Abstract interface for the Fluid Memory service.

    Implement this to swap between local in-process execution
    and a standalone remote service without changing call sites.
    """

    async def touch(self, event: FluidUpdateEvent) -> None:
        """Process a touch event after an episodic memory write."""
        ...

    async def get_state(self, node_id: str) -> Optional[FluidMemoryState]:
        """Get the current fluid state for a single node."""
        ...

    async def get_states(self, node_ids: List[str]) -> List[FluidMemoryState]:
        """Get fluid states for a batch of nodes."""
        ...

    async def apply_fluid_scores(self, bundles: List) -> List:
        """Apply fluid score adjustments to retrieval bundles in-place."""
        ...


# ---------------------------------------------------------------------------
# Local implementation (wraps FluidMemoryEngine)
# ---------------------------------------------------------------------------

class LocalFluidMemoryService:
    """
    Local in-process implementation of FluidMemoryServiceInterface.

    Delegates to FluidMemoryEngine for touch/state operations and
    FluidStateStore + fluid_score() for retrieval adjustments.
    """

    def __init__(
        self,
        graph_engine: Optional["GraphProvider"],
        store: "FluidStateStore",
    ) -> None:
        self._graph = graph_engine
        self._store = store
        self._engine = None  # lazily initialised to avoid import cycles

    def _get_engine(self):
        if self._engine is None:
            from m_flow.memory.fluid.engine import FluidMemoryEngine
            self._engine = FluidMemoryEngine(self._graph, self._store)
        return self._engine

    async def touch(self, event: FluidUpdateEvent) -> None:
        await self._get_engine().touch(event)

    async def get_state(self, node_id: str) -> Optional[FluidMemoryState]:
        return await self._get_engine().get_state(node_id)

    async def get_states(self, node_ids: List[str]) -> List[FluidMemoryState]:
        return await self._get_engine().get_states(node_ids)

    async def apply_fluid_scores(self, bundles: List) -> List:
        """
        Adjust retrieval bundle scores using fluid state.

        Implements the full v2 scoring contract:
        1. Normalize distance scores to semantic similarity [0, 1]
        2. Compute graph_score from best_path in bundle
        3. Call compute_effective_score(semantic, state, graph)
        4. Convert back to distance score (lower-is-better)
        5. Store explanation fields on bundle

        Skips bundles with no active fluid state (activation == 0.0).
        Failures are logged and original bundles returned intact.
        """
        from m_flow.shared.logging_utils import get_logger
        from m_flow.memory.fluid.scoring import (
            compute_effective_score, explain_effective_score,
        )
        from m_flow.memory.fluid.config import get_fluid_config

        logger = get_logger("fluid.service")
        cfg = get_fluid_config()

        if not bundles:
            return bundles

        # Snapshot original scores for rollback on failure
        original_scores = {b.episode_id: b.score for b in bundles}

        try:
            # Collect all distances for normalization
            episode_ids = [b.episode_id for b in bundles]
            states = await self._store.get_many(episode_ids)
            state_map = {s.node_id: s for s in states if s.activation > 0.0}

            # Find min/max distances for normalization
            distances = [b.score for b in bundles if hasattr(b, "score")]
            if not distances:
                return bundles

            min_dist = min(distances)
            max_dist = max(distances)
            dist_range = max_dist - min_dist if max_dist > min_dist else 1.0

            for bundle in bundles:
                # Store base distance score
                if not hasattr(bundle, "base_distance_score"):
                    bundle.base_distance_score = bundle.score

                state = state_map.get(bundle.episode_id)
                if not state:
                    continue

                # Normalize distance to semantic similarity [0, 1]
                # Lower distance = higher similarity
                if dist_range > 0:
                    semantic_score = 1.0 - ((bundle.score - min_dist) / dist_range)
                else:
                    semantic_score = 0.5  # All equal distances

                # Compute graph_score from best_path if available
                graph_score = self._compute_graph_score(bundle)

                # Calculate effective score
                effective = compute_effective_score(
                    semantic_score=semantic_score,
                    state=state,
                    graph_score=graph_score,
                )

                # Store scoring metadata
                bundle.semantic_score = semantic_score
                bundle.graph_score = graph_score
                bundle.fluid_effective_score = effective
                bundle.fluid_score_explanation = explain_effective_score(
                    semantic_score=semantic_score,
                    state=state,
                    graph_score=graph_score,
                )

                # Convert back to lower-is-better distance score
                # effective is [0, 1] higher=better, so distance = 1 - effective
                bundle.final_distance_score = 1.0 - effective
                bundle.score = bundle.final_distance_score

        except Exception as exc:
            logger.warning(
                "fluid.service: apply_fluid_scores failed: %s (%s)",
                type(exc).__name__,
                exc,
            )
            if cfg.fail_closed_on_scoring_error:
                # Rollback to original scores
                for bundle in bundles:
                    original = original_scores.get(bundle.episode_id)
                    if original is not None:
                        bundle.score = original
                        # Clear partial fluid fields
                        bundle.semantic_score = None
                        bundle.graph_score = None
                        bundle.fluid_effective_score = None
                        bundle.final_distance_score = None
                        bundle.fluid_score_explanation = None
                return bundles

        return bundles

    def _compute_graph_score(self, bundle) -> float:
        """
        Compute graph proximity score from bundle's best_path.

        best_path is a string enum value, not a list.
        Scores:
            direct_episode -> 1.00
            facet -> 0.85
            facet_entity -> 0.78
            point -> 0.72
            entity -> 0.65
            unknown -> 0.50
        """
        path = str(getattr(bundle, "best_path", "")).lower()

        if path == "direct_episode":
            return 1.00
        if path == "facet":
            return 0.85
        if path == "facet_entity":
            return 0.78
        if path == "point":
            return 0.72
        if path == "entity":
            return 0.65

        return 0.50  # unknown/default


# ---------------------------------------------------------------------------
# Remote stub (for future extraction)
# ---------------------------------------------------------------------------

class RemoteFluidMemoryService:
    """
    Stub for a future remote Fluid Memory service.

    When the fluid layer is extracted into its own process, replace
    LocalFluidMemoryService with this class and point base_url at
    the deployed service endpoint.

    Not yet implemented — raises NotImplementedError on all calls.
    """

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url

    async def touch(self, event: FluidUpdateEvent) -> None:
        raise NotImplementedError("RemoteFluidMemoryService is not yet implemented")

    async def get_state(self, node_id: str) -> Optional[FluidMemoryState]:
        raise NotImplementedError("RemoteFluidMemoryService is not yet implemented")

    async def get_states(self, node_ids: List[str]) -> List[FluidMemoryState]:
        raise NotImplementedError("RemoteFluidMemoryService is not yet implemented")

    async def apply_fluid_scores(self, bundles: List) -> List:
        raise NotImplementedError("RemoteFluidMemoryService is not yet implemented")
