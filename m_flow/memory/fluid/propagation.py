"""
Fluid Memory Propagation

Handles activation ripple through the graph.
When a node is touched, activation propagates to connected nodes.
"""

from typing import Dict, List, Set, Tuple, Optional
from collections import deque

from m_flow.adapters.graph.graph_db_interface import GraphProvider


async def get_connected_nodes(
    graph_engine: GraphProvider,
    node_id: str,
    edge_types: Optional[List[str]] = None,
) -> List[Tuple[str, str, float]]:
    """
    Get nodes connected to a given node.
    
    Args:
        graph_engine: Graph database provider
        node_id: Starting node ID
        edge_types: Optional list of edge types to follow (None = all)
        
    Returns:
        List of tuples (connected_node_id, edge_type, edge_weight)
    """
    # Query for connected nodes via Cypher
    if edge_types:
        type_filter = " OR ".join([f"type(r) = '{t}'" for t in edge_types])
        cypher = f"""
            MATCH (n)-[r]-(m)
            WHERE n.id = $node_id AND ({type_filter})
            RETURN m.id as connected_id, type(r) as edge_type, 
                   coalesce(r.weight, 1.0) as weight
        """
    else:
        cypher = """
            MATCH (n)-[r]-(m)
            WHERE n.id = $node_id
            RETURN m.id as connected_id, type(r) as edge_type,
                   coalesce(r.weight, 1.0) as weight
        """
    
    result = await graph_engine.query(cypher, {"node_id": node_id})
    
    connections = []
    for row in result:
        connected_id = row.get("connected_id") or row.get("m.id")
        edge_type = row.get("edge_type") or row.get("type(r)")
        weight = row.get("weight") or row.get("coalesce(r.weight, 1.0)") or 1.0
        if connected_id:
            connections.append((str(connected_id), str(edge_type), float(weight)))
    
    return connections


async def propagate_activation(
    graph_engine: GraphProvider,
    seed_node_ids: List[str],
    start_activation: float = 0.18,
    max_depth: int = 2,
    edge_types: Optional[List[str]] = None,
    decay_per_hop: float = 0.5,
    min_activation: float = 0.05,
) -> Dict[str, float]:
    """
    Propagate activation from seed nodes through the graph.
    
    Uses BFS to ripple activation through connected nodes.
    Activation decays with each hop (default 50% per hop).
    
    Args:
        graph_engine: Graph database provider
        seed_node_ids: Starting nodes to propagate from
        start_activation: Activation level to propagate from seeds
        max_depth: Maximum propagation depth (default 2)
        edge_types: Optional edge types to follow (None = all types)
        decay_per_hop: Activation decay factor per hop (0.5 = 50%)
        min_activation: Minimum activation to continue propagating
        
    Returns:
        Dict mapping node_id -> propagated activation value
    """
    # Track activation levels and visited depths
    activation_levels: Dict[str, float] = {}
    visited: Dict[str, int] = {}  # node_id -> depth visited
    
    # BFS queue: (node_id, depth, activation)
    queue: deque[Tuple[str, int, float]] = deque()
    
    # Initialize with seed nodes
    for node_id in seed_node_ids:
        activation_levels[node_id] = start_activation
        visited[node_id] = 0
        queue.append((node_id, 0, start_activation))
    
    # BFS propagation
    while queue:
        current_id, depth, current_activation = queue.popleft()
        
        # Stop if we've reached max depth
        if depth >= max_depth:
            continue
        
        # Get connected nodes
        try:
            connections = await get_connected_nodes(
                graph_engine, current_id, edge_types
            )
        except Exception:
            # If graph query fails, skip this node
            continue
        
        # Propagate to each connected node
        for connected_id, edge_type, edge_weight in connections:
            # Calculate propagated activation
            propagated = current_activation * decay_per_hop * edge_weight
            
            # Skip if below minimum
            if propagated < min_activation:
                continue
            
            # Update activation (max of existing and propagated)
            if connected_id in activation_levels:
                activation_levels[connected_id] = max(
                    activation_levels[connected_id], 
                    propagated
                )
            else:
                activation_levels[connected_id] = propagated
            
            # Add to queue if not visited at this depth or deeper
            if connected_id not in visited or visited[connected_id] > depth + 1:
                visited[connected_id] = depth + 1
                queue.append((connected_id, depth + 1, propagated))
    
    return activation_levels


# Default episodic edge types for propagation
EPISODIC_EDGE_TYPES = [
    "has_facet",
    "has_point", 
    "involves_entity",
    "supported_by",
    "includes_chunk",
    "same_entity_as",
]


async def propagate_activation_episodic(
    graph_engine: GraphProvider,
    seed_node_ids: List[str],
    start_activation: float = 0.18,
    max_depth: int = 2,
) -> Dict[str, float]:
    """
    Propagate activation through episodic memory graph.
    
    Uses standard episodic edge types (has_facet, involves_entity, etc.).
    
    Args:
        graph_engine: Graph database provider
        seed_node_ids: Starting nodes to propagate from
        start_activation: Activation level to propagate from seeds
        max_depth: Maximum propagation depth (default 2)
        
    Returns:
        Dict mapping node_id -> propagated activation value
    """
    return await propagate_activation(
        graph_engine=graph_engine,
        seed_node_ids=seed_node_ids,
        start_activation=start_activation,
        max_depth=max_depth,
        edge_types=EPISODIC_EDGE_TYPES,
        decay_per_hop=0.5,
        min_activation=0.05,
    )
