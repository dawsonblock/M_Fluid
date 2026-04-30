"""
Fluid Memory Engine

Main orchestrator for the fluid memory system.
Runs after episodic writes and before retrieval.
"""

from time import time
from typing import TYPE_CHECKING, Optional, List

from m_flow.memory.fluid.models import FluidUpdateEvent, FluidMemoryState
from m_flow.memory.fluid.state_store import FluidStateStore
from m_flow.memory.fluid.propagation import propagate_activation_episodic
from m_flow.memory.fluid.decay import apply_decay, update_recency_scores
from m_flow.memory.fluid.contradiction import apply_contradictions
from m_flow.memory.fluid.audit import FluidAuditLogger, AuditEventType

if TYPE_CHECKING:
    from m_flow.adapters.graph.graph_db_interface import GraphProvider


class FluidMemoryEngine:
    """
    Fluid Memory Engine - orchestrates fluid state updates.
    
    The engine receives touch events from episodic memory writes,
    updates fluid state, applies decay and contradictions,
    and propagates activation through the graph.
    
    Usage:
        engine = FluidMemoryEngine(graph_engine, store)
        await engine.touch(FluidUpdateEvent(
            touched_node_ids=[episode.id, facet.id, entity.id],
            source_id=document_id,
            source_type="mainstream_news",
            source_trust=0.60,
            salience=0.7,
            legal_weight=0.3,
        ))
    """
    
    def __init__(
        self,
        graph_engine: "GraphProvider",
        store: FluidStateStore,
        enable_audit: bool = True,
    ):
        self.graph = graph_engine
        self.store = store
        self.audit = FluidAuditLogger(store) if enable_audit else None
        
        # Configuration
        self.activation_increment = 0.25
        self.max_activation = 1.0
        self.propagation_start_activation = 0.18
        self.propagation_max_depth = 2
    
    async def touch(self, event: FluidUpdateEvent) -> None:
        """
        Process a touch event - the main entry point.
        
        A touch event occurs when episodic memory is written.
        It triggers:
        1. Activation boost for touched nodes
        2. Field updates (trust, salience, legal_weight)
        3. Decay application
        4. Contradiction processing
        5. Activation propagation to connected nodes
        
        Args:
            event: The fluid update event
        """
        now = time()
        
        # Step 1: Get current states for touched nodes
        states = await self.store.get_many(event.touched_node_ids)
        
        # Step 2: Update touched nodes
        for state in states:
            # Boost activation (capped at max)
            state.activation = min(
                self.max_activation,
                state.activation + self.activation_increment
            )
            
            # Increment reinforcement count
            state.reinforcement_count += 1
            
            # Update trust (take max of current and event)
            state.source_trust = max(state.source_trust, event.source_trust)
            
            # Update salience (take max)
            state.salience = max(state.salience, event.salience)
            
            # Update legal weight (take max)
            state.legal_weight = max(state.legal_weight, event.legal_weight)
            
            # Update timestamp
            state.last_touched_at = now
            
            # Recency is always 1.0 when just touched
            state.recency_score = 1.0
        
        # Step 3: Apply decay to all touched states
        states = apply_decay(states, now)
        
        # Step 4: Apply contradiction pressure
        states = apply_contradictions(states, event)
        
        # Step 5: Persist updated states
        await self.store.upsert_many(states)
        
        # Step 6: Audit log
        if self.audit:
            for state in states:
                await self.audit.log_touch(
                    node_id=state.node_id,
                    new_activation=state.activation,
                    source_id=event.source_id,
                    source_type=event.source_type,
                )
        
        # Step 7: Propagate activation through graph
        ripple_updates = await propagate_activation_episodic(
            graph_engine=self.graph,
            seed_node_ids=event.touched_node_ids,
            start_activation=self.propagation_start_activation,
            max_depth=self.propagation_max_depth,
        )
        
        # Step 8: Merge propagated activation into store
        if ripple_updates:
            merge_results = await self.store.merge_activation(
                ripple_updates, 
                merge_mode="max"
            )
            
            # Audit log propagation
            if self.audit:
                for result in merge_results:
                    if result.merged and result.new_activation > result.old_activation:
                        await self.audit.log_propagation(
                            node_id=result.node_id,
                            old_activation=result.old_activation,
                            new_activation=result.new_activation,
                            source_nodes=event.touched_node_ids,
                        )
    
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
        Manually activate a node (for admin/debug purposes).
        
        Args:
            node_id: Node to activate
            activation_amount: Amount to add to activation
            reason: Reason for manual activation (for audit)
        """
        state = await self.store.get(node_id)
        if not state:
            state = FluidMemoryState(node_id=node_id)
        
        old_activation = state.activation
        state.activation = min(1.0, state.activation + activation_amount)
        state.last_touched_at = time()
        
        await self.store.upsert(state)
        
        if self.audit:
            await self.audit.log_event(
                event_type=AuditEventType.MANUAL,
                node_id=node_id,
                field_changed="activation",
                old_value=old_activation,
                new_value=state.activation,
                metadata={"reason": reason},
            )
    
    async def decay_all(self, threshold: float = 0.01) -> int:
        """
        Apply decay to all active nodes.
        
        Typically called by a background job.
        
        Args:
            threshold: Only decay nodes above this activation
            
        Returns:
            Number of nodes decayed
        """
        states = await self.store.get_all_with_activation_above(threshold)
        
        if not states:
            return 0
        
        now = time()
        decayed_states = apply_decay(states, now)
        decayed_states = update_recency_scores(decayed_states, now)
        
        await self.store.upsert_many(decayed_states)
        
        # Audit log
        if self.audit:
            for state in decayed_states:
                await self.audit.log_decay(
                    node_id=state.node_id,
                    new_activation=state.activation,
                )
        
        return len(decayed_states)
    
    async def get_active_nodes(
        self,
        min_activation: float = 0.1,
    ) -> List[FluidMemoryState]:
        """
        Get all currently active nodes.
        
        Args:
            min_activation: Minimum activation threshold
            
        Returns:
            List of active fluid states
        """
        return await self.store.get_all_with_activation_above(min_activation)
