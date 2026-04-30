"""
Fluid Memory Contradiction

Handles contradiction pressure between competing sources.
When sources contradict, pressure builds on affected nodes.
"""

from typing import List, Set, Dict
from m_flow.memory.fluid.models import FluidMemoryState, FluidUpdateEvent


def compute_contradiction_pressure(
    node_id: str,
    event: FluidUpdateEvent,
    base_pressure: float = 0.15,
) -> float:
    """
    Compute contradiction pressure for a node from an event.
    
    Pressure increases when:
    - This source contradicts the node
    - The source has high trust (confident contradictions hurt more)
    
    Args:
        node_id: The node to compute pressure for
        event: The update event carrying contradiction info
        base_pressure: Base pressure value per contradiction
        
    Returns:
        Pressure value [0-1] to add to node's contradiction_pressure
    """
    pressure = 0.0
    
    # Check if this event contradicts the node
    if node_id in event.contradicts:
        # More trusted sources create more pressure when they contradict
        pressure += base_pressure * event.source_trust
    
    return pressure


def apply_contradictions(
    states: List[FluidMemoryState],
    event: FluidUpdateEvent,
    decay_factor: float = 0.9,
) -> List[FluidMemoryState]:
    """
    Apply contradiction pressure to states.
    
    Contradiction pressure:
    - Increases when sources contradict a node
    - Decays over time (old contradictions fade)
    - Reduces retrieval scores (negatively weighted)
    
    Args:
        states: List of fluid states to update
        event: The update event with contradiction info
        decay_factor: How much existing pressure decays (0.9 = 10% decay)
        
    Returns:
        The same list with updated contradiction_pressure values
    """
    contradicted_set: Set[str] = set(event.contradicts)
    
    for state in states:
        # Decay existing pressure
        state.contradiction_pressure *= decay_factor
        
        # Add new pressure if this node is contradicted
        if state.node_id in contradicted_set:
            new_pressure = compute_contradiction_pressure(state.node_id, event)
            state.contradiction_pressure = min(1.0, state.contradiction_pressure + new_pressure)
    
    return states


def resolve_contradictions(
    states: List[FluidMemoryState],
    supported_nodes: List[str],
    contradicted_nodes: List[str],
    source_trust: float,
) -> List[FluidMemoryState]:
    """
    Batch resolve contradictions from a source.
    
    Updates confidence and contradiction_pressure based on
    which nodes the source supports vs contradicts.
    
    Args:
        states: States to potentially update
        supported_nodes: Nodes this source supports
        contradicted_nodes: Nodes this source contradicts
        source_trust: Trust level of the source
        
    Returns:
        Updated states
    """
    supported_set: Set[str] = set(supported_nodes)
    contradicted_set: Set[str] = set(contradicted_nodes)
    
    for state in states:
        if state.node_id in supported_set:
            # Supporting a node increases confidence
            # Higher trust sources boost confidence more
            confidence_boost = 0.1 * source_trust
            state.confidence = min(1.0, state.confidence + confidence_boost)
            
            # Supporting also reduces contradiction pressure
            state.contradiction_pressure *= 0.8
            
        elif state.node_id in contradicted_set:
            # Contradicting a node adds pressure
            pressure_increase = 0.15 * source_trust
            state.contradiction_pressure = min(1.0, state.contradiction_pressure + pressure_increase)
    
    return states


def get_contradiction_summary(
    states: List[FluidMemoryState],
) -> Dict[str, any]:
    """
    Get summary statistics of contradiction pressure across states.
    
    Args:
        states: List of states to analyze
        
    Returns:
        Dict with summary statistics
    """
    if not states:
        return {
            "total_nodes": 0,
            "nodes_with_pressure": 0,
            "max_pressure": 0.0,
            "avg_pressure": 0.0,
        }
    
    pressures = [s.contradiction_pressure for s in states]
    nodes_with_pressure = sum(1 for p in pressures if p > 0.01)
    
    return {
        "total_nodes": len(states),
        "nodes_with_pressure": nodes_with_pressure,
        "max_pressure": max(pressures),
        "avg_pressure": sum(pressures) / len(pressures),
    }
