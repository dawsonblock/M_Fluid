"""
Fluid Memory Module for M-flow

A mutable operational state layer that runs after episodic memory writes
and before retrieval scoring, creating a "water effect" where touching
one node ripples activation through connected memory.

Usage:
    from m_flow.memory.fluid import (
        FluidMemoryEngine,
        FluidMemoryState,
        FluidUpdateEvent,
        fluid_score,
        get_source_weights,
    )
    
    # Create engine
    engine = FluidMemoryEngine(graph_engine, store)
    
    # Touch after episodic write
    await engine.touch(FluidUpdateEvent(
        touched_node_ids=[episode.id, facet.id, entity.id],
        source_id=document_id,
        source_type="mainstream_news",
        source_trust=0.60,
        salience=0.7,
    ))
    
    # Score in retrieval
    final_score = fluid_score(base_score, fluid_state)

Design Principles:
- Raw evidence never changes
- Graph links can change carefully
- Fluid state changes constantly
- Summaries can be regenerated from evidence + current state
"""

from m_flow.memory.fluid.models import (
    FluidMemoryState,
    FluidUpdateEvent,
    ClaimConflict,
    SourceLineageRecord,
    MediaAmplificationEvent,
    get_source_weights,
)
from m_flow.memory.fluid.scoring import (
    compute_effective_score,
    explain_effective_score,
    fluid_score,
    fluid_score_similarity,
    compute_fluid_boost,
    explain_fluid_score,
    should_boost_retrieval,
)
from m_flow.memory.fluid.citation_graph import CitationGraph
from m_flow.memory.fluid.timeline import TimelineCompressor, TimelineEvent
from m_flow.memory.fluid.jurisdiction import JurisdictionWeighter, get_jurisdiction_weighter
from m_flow.memory.fluid.graph_access import (
    row_get,
    get_node_text,
    get_neighbour_ids,
    get_connected_nodes,
)
from m_flow.memory.fluid.state_store import (
    FluidStateStore,
    ActivationMergeUpdate,
)
from m_flow.memory.fluid.engine import FluidMemoryEngine
from m_flow.memory.fluid.audit import (
    FluidAuditLogger,
    FluidProvenance,
    AuditEventType,
)
from m_flow.memory.fluid.config import FluidMemoryConfig, get_fluid_config
from m_flow.memory.fluid.service_interface import (
    FluidMemoryServiceInterface,
    LocalFluidMemoryService,
    RemoteFluidMemoryService,
)

__all__ = [
    # Models
    "FluidMemoryState",
    "FluidUpdateEvent",
    "ClaimConflict",
    "SourceLineageRecord",
    "MediaAmplificationEvent",
    "ActivationMergeUpdate",

    # Source weights (sync shim)
    "get_source_weights",

    # Scoring
    "fluid_score",
    "fluid_score_similarity",
    "compute_fluid_boost",
    "explain_fluid_score",
    "should_boost_retrieval",
    "compute_effective_score",
    "explain_effective_score",

    # JudgeTracker / legal features
    "CitationGraph",
    "TimelineCompressor",
    "TimelineEvent",
    "JurisdictionWeighter",
    "get_jurisdiction_weighter",

    # Graph access helpers
    "row_get",
    "get_node_text",
    "get_neighbour_ids",
    "get_connected_nodes",

    # Engine and Store
    "FluidMemoryEngine",
    "FluidStateStore",

    # Audit + Provenance
    "FluidAuditLogger",
    "FluidProvenance",
    "AuditEventType",

    # Config
    "FluidMemoryConfig",
    "get_fluid_config",

    # Service interface
    "FluidMemoryServiceInterface",
    "LocalFluidMemoryService",
    "RemoteFluidMemoryService",
]
