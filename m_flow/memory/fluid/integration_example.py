"""
Fluid Memory Integration Example

Shows how to integrate the fluid memory engine into M-flow's
episodic memory pipeline and retrieval scoring.

This is example code - actual integration requires modifying:
1. m_flow/memory/episodic/write_episodic_memories.py
2. m_flow/retrieval/episodic/bundle_scorer.py
"""

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from m_flow.core.domain.models import Episode, Facet, Entity
    from m_flow.adapters.graph.graph_db_interface import GraphProvider
    from m_flow.adapters.vector.vector_db_interface import VectorProvider

from m_flow.memory.fluid import (
    FluidMemoryEngine,
    FluidStateStore,
    FluidUpdateEvent,
    fluid_score,
    get_source_weights,
)


class FluidMemoryIntegration:
    """
    Helper class for integrating fluid memory into M-flow.
    
    Usage:
        integration = FluidMemoryIntegration(graph_engine)
        
        # After episodic write:
        await integration.touch_after_episodic_write(
            episode=episode,
            facets=facets,
            entities=entities,
            document_id=doc_id,
            source_type="mainstream_news",
        )
        
        # In retrieval:
        adjusted_score = await integration.adjust_retrieval_score(
            base_score, episode_id
        )
    """
    
    def __init__(
        self,
        graph_engine: "GraphProvider",
        db_provider: str = "sqlite",
        db_path: str = "",
    ):
        self.store = FluidStateStore(
            db_provider=db_provider,
            db_path=db_path,
            db_name="fluid_memory",
        )
        self.engine = FluidMemoryEngine(graph_engine, self.store)
    
    async def touch_after_episodic_write(
        self,
        episode: "Episode",
        facets: List["Facet"],
        entities: List["Entity"],
        document_id: str,
        source_type: str = "unknown",
        custom_salience: float = 0.5,
    ) -> None:
        """
        Call this after creating/updating episodic memory nodes.
        
        This is the main integration point - triggers fluid state updates
        and activation propagation through the graph.
        
        Args:
            episode: The episode node that was written
            facets: Facet nodes that were written
            entities: Entity nodes that were written
            document_id: Source document identifier
            source_type: Type of source (court_record, police_release, etc.)
            custom_salience: Override salience (default: computed from source)
        """
        # Get source weights
        trust, legal = get_source_weights(source_type)
        
        # Collect all touched node IDs
        touched_ids = [episode.id]
        touched_ids.extend(f.id for f in facets)
        touched_ids.extend(e.id for e in entities)
        
        # Create update event
        event = FluidUpdateEvent(
            touched_node_ids=touched_ids,
            source_id=document_id,
            source_type=source_type,
            source_trust=trust,
            salience=custom_salience,
            legal_weight=legal,
        )
        
        # Process through fluid engine
        await self.engine.touch(event)
    
    async def adjust_retrieval_score(
        self,
        base_score: float,
        episode_id: str,
    ) -> float:
        """
        Adjust a retrieval score using fluid state.
        
        Call this in bundle_scorer.py after computing base episode score.
        
        Args:
            base_score: Base retrieval score (lower is better)
            episode_id: Episode being scored
            
        Returns:
            Adjusted score with fluid boost/penalty applied
        """
        state = await self.engine.get_state(episode_id)
        if not state:
            return base_score
        
        return fluid_score(base_score, state)
    
    async def get_active_context(
        self,
        min_activation: float = 0.2,
    ) -> List[str]:
        """
        Get list of currently active node IDs.
        
        Useful for "what's currently on the system's mind" features.
        
        Args:
            min_activation: Minimum activation threshold
            
        Returns:
            List of active node IDs
        """
        states = await self.engine.get_active_nodes(min_activation)
        return [s.node_id for s in states]


# =============================================================================
# Integration Code for write_episodic_memories.py
# =============================================================================

"""
Add to write_episodic_memories.py after episode nodes are created:

    # === FLUID MEMORY INTEGRATION ===
    from m_flow.memory.fluid import FluidMemoryEngine, FluidStateStore, FluidUpdateEvent, get_source_weights
    
    # Create fluid engine (could be cached/passed in context)
    fluid_store = FluidStateStore(db_provider="sqlite", db_path=...)
    fluid_engine = FluidMemoryEngine(graph_engine, fluid_store)
    
    # Collect node IDs from created episode
    touched_ids = [episode.id]
    facet_ids = []
    entity_ids = []
    
    for edge, facet in (episode.has_facet or []):
        touched_ids.append(facet.id)
        facet_ids.append(facet.id)
    
    for edge, entity in (episode.involves_entity or []):
        touched_ids.append(entity.id)
        entity_ids.append(entity.id)
    
    # Get source weights based on document type
    trust, legal = get_source_weights(document.source_type)
    
    # Touch fluid memory
    await fluid_engine.touch(FluidUpdateEvent(
        touched_node_ids=touched_ids,
        source_id=document.id,
        source_type=document.source_type or "unknown",
        source_trust=trust,
        salience=0.7,  # Could be computed from document importance
        legal_weight=legal,
    ))
    # === END FLUID MEMORY ===
"""


# =============================================================================
# Integration Code for bundle_scorer.py
# =============================================================================

"""
Add to bundle_scorer.py in compute_episode_bundles or scoring loop:

    # === FLUID MEMORY INTEGRATION ===
    from m_flow.memory.fluid import FluidStateStore, fluid_score
    
    # Create store (could be cached)
    fluid_store = FluidStateStore(db_provider="sqlite", db_path=...)
    
    # After computing base bundle score, adjust with fluid
    for bundle in bundles:
        fluid_state = await fluid_store.get(bundle.episode_id)
        if fluid_state:
            bundle.score = fluid_score(bundle.score, fluid_state)
    # === END FLUID MEMORY ===
"""


# =============================================================================
# Factory for creating fluid engine from M-flow context
# =============================================================================

async def create_fluid_engine_from_context() -> FluidMemoryEngine:
    """
    Factory to create fluid engine using M-flow's standard configuration.
    
    This would integrate with M-flow's config system.
    """
    from m_flow.adapters.graph import get_graph_provider
    from m_flow.base_config import get_base_config
    
    # Get graph engine
    graph_engine = await get_graph_provider()
    
    # Get storage config
    base_cfg = get_base_config()
    db_path = base_cfg.system_root_directory / "databases"
    
    # Create store
    store = FluidStateStore(
        db_provider="sqlite",
        db_path=str(db_path),
        db_name="fluid_memory",
    )
    
    return FluidMemoryEngine(graph_engine, store)
