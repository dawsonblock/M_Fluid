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

        Reads states for all bundle episode IDs from the store and
        applies bounded fluid_score() to each bundle with a state.
        Silently skips bundles with no fluid state.
        """
        from m_flow.shared.logging_utils import get_logger
        from m_flow.memory.fluid.scoring import fluid_score

        logger = get_logger("fluid.service")

        try:
            episode_ids = [b.episode_id for b in bundles]
            states = await self._store.get_many(episode_ids)
            state_map = {s.node_id: s for s in states if s.activation > 0.0}

            for bundle in bundles:
                state = state_map.get(bundle.episode_id)
                if state:
                    bundle.score = fluid_score(bundle.score, state)

        except Exception as exc:
            logger.warning("fluid.service: apply_fluid_scores failed: %s", exc)

        return bundles


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
