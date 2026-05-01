"""
Fluid Memory Graph Access Layer

Abstracts graph provider differences (dict rows vs tuple rows).
Provides safe, logged access to graph data for fluid memory operations.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, List, Optional, Tuple

from m_flow.shared.logging_utils import get_logger

logger = get_logger("fluid.graph_access")


def row_get(
    row: Any,
    names: List[str],
    index: int | None = None,
    default: Any = None,
) -> Any:
    """
    Safely extract a value from a graph query row.

    Supports:
    - dict / Mapping rows: uses names[0] as key
    - tuple/list rows: uses index position
    - object rows with attributes: tries getattr

    Args:
        row: The query result row (dict, tuple, or object)
        names: List of possible column/field names to try
        index: Positional index for tuple rows
        default: Value to return if extraction fails

    Returns:
        The extracted value or default
    """
    if row is None:
        return default

    # Try dict-style access first
    if isinstance(row, Mapping):
        for name in names:
            if name in row:
                val = row[name]
                return val if val is not None else default
        return default

    # Try tuple/list-style access by index
    if index is not None and isinstance(row, (list, tuple)):
        if 0 <= index < len(row):
            val = row[index]
            return val if val is not None else default
        return default

    # Try attribute access
    for name in names:
        if hasattr(row, name):
            val = getattr(row, name)
            return val if val is not None else default

    return default


async def safe_query(
    graph_engine: Any,
    query: str,
    params: Optional[dict] = None,
) -> Any:
    """
    Execute a query with parameterized inputs where supported.

    Tries parameterized query first (preferred for security).
    Falls back to f-string interpolation if provider doesn't support parameters.

    Args:
        graph_engine: The graph provider/engine
        query: Cypher query with $param placeholders
        params: Dictionary of parameters

    Returns:
        Query result
    """
    if params:
        try:
            # Try parameterized query (preferred for security)
            return await graph_engine.query(query, params)
        except (TypeError, AttributeError) as exc:
            # Provider may not support parameterized queries
            # Fall back to safe interpolation for known-safe values
            logger.debug(
                "fluid.graph_access: parameterized query not supported, "
                "falling back to interpolation: %s",
                type(exc).__name__,
            )
            # For node_id and similar internal IDs, interpolation is acceptable
            # as they are not user-supplied
            interpolated = query
            for key, value in params.items():
                interpolated = interpolated.replace(f"${key}", f"'{value}'")
            return await graph_engine.query(interpolated)
    return await graph_engine.query(query)


async def get_node_text(graph_engine: Any, node_id: str) -> Optional[str]:
    """
    Fetch text content for a node from the graph.

    Args:
        graph_engine: The graph provider/engine
        node_id: Node identifier

    Returns:
        Node text or None if not found/error
    """
    provider_name = _get_provider_name(graph_engine)
    try:
        # Try different query patterns common in M-flow
        queries = [
            "MATCH (n) WHERE n.id = $node_id RETURN n.text as text",
            "MATCH (n {id: $node_id}) RETURN n.text as text",
        ]

        for query in queries:
            try:
                result = await safe_query(graph_engine, query, {"node_id": node_id})
                if result:
                    rows = list(result)
                    if rows:
                        text = row_get(rows[0], ["text", "n.text"], index=0, default=None)
                        if text:
                            return str(text).strip()
            except Exception:
                continue

        # Fallback: try to get any string property
        try:
            props_query = "MATCH (n) WHERE n.id = $node_id RETURN properties(n) as props"
            result = await safe_query(graph_engine, props_query, {"node_id": node_id})
            if result:
                rows = list(result)
                if rows:
                    props = row_get(rows[0], ["props", "properties(n)"], index=0, default={})
                    if isinstance(props, Mapping):
                        for key in ["text", "content", "summary", "name", "title"]:
                            if key in props and props[key]:
                                return str(props[key]).strip()
        except Exception:
            pass

    except Exception as exc:
        logger.warning(
            "fluid.graph_access: failed to fetch text for node %s (provider=%s): %s (%s)",
            node_id,
            provider_name,
            type(exc).__name__,
            exc,
        )

    return None


async def get_neighbour_ids(
    graph_engine: Any,
    node_id: str,
    limit: int = 5,
) -> List[str]:
    """
    Get IDs of neighboring nodes (connected by any edge).

    Args:
        graph_engine: The graph provider/engine
        node_id: Center node identifier
        limit: Maximum neighbors to return

    Returns:
        List of neighbor node IDs
    """
    neighbor_ids: List[str] = []
    provider_name = _get_provider_name(graph_engine)

    try:
        # Try different edge patterns common in M-flow
        edge_types = ["has_facet", "involves_entity", "has_point", "supported_by", "includes_chunk"]

        for edge_type in edge_types:
            query = f"""
                MATCH (n)-[:{edge_type}]-(m)
                WHERE n.id = $node_id
                RETURN m.id as neighbor_id
                LIMIT {limit}
            """
            try:
                result = await safe_query(graph_engine, query, {"node_id": node_id})
                if result:
                    for row in result:
                        nid = row_get(row, ["neighbor_id", "m.id"], index=0, default=None)
                        if nid and nid not in neighbor_ids:
                            neighbor_ids.append(str(nid))
                            if len(neighbor_ids) >= limit:
                                return neighbor_ids
            except Exception:
                continue

        # Fallback: generic neighbor query
        if not neighbor_ids:
            generic_query = """
                MATCH (n)--(m)
                WHERE n.id = $node_id
                RETURN m.id as neighbor_id
                LIMIT $limit
            """
            try:
                result = await safe_query(graph_engine, generic_query, {"node_id": node_id, "limit": limit})
                if result:
                    for row in result:
                        nid = row_get(row, ["neighbor_id", "m.id"], index=0, default=None)
                        if nid and nid not in neighbor_ids:
                            neighbor_ids.append(str(nid))
                            if len(neighbor_ids) >= limit:
                                break
            except Exception as exc:
                logger.debug(
                    "fluid.graph_access: generic neighbor query failed for %s (provider=%s): %s",
                    node_id,
                    provider_name,
                    exc,
                )

    except Exception as exc:
        logger.warning(
            "fluid.graph_access: failed to get neighbors for node %s (provider=%s): %s (%s)",
            node_id,
            provider_name,
            type(exc).__name__,
            exc,
        )

    return neighbor_ids


async def get_connected_nodes(
    graph_engine: Any,
    node_id: str,
    edge_types: Optional[List[str]] = None,
) -> List[Tuple[str, str, float]]:
    """
    Get connected nodes with edge type and relationship strength.

    Args:
        graph_engine: The graph provider/engine
        node_id: Center node identifier
        edge_types: Optional list of edge types to filter (None = all)

    Returns:
        List of (neighbor_id, edge_type, strength) tuples
        Strength is a float [0-1] indicating connection weight
    """
    connections: List[Tuple[str, str, float]] = []
    provider_name = _get_provider_name(graph_engine)

    try:
        types_to_query = edge_types if edge_types else ["has_facet", "involves_entity", "has_point", "supported_by"]

        for edge_type in types_to_query:
            query = f"""
                MATCH (n)-[r:{edge_type}]-(m)
                WHERE n.id = $node_id
                RETURN m.id as neighbor_id, type(r) as edge_type, r.weight as weight
            """
            try:
                result = await safe_query(graph_engine, query, {"node_id": node_id})
                if result:
                    for row in result:
                        nid = row_get(row, ["neighbor_id", "m.id"], index=0, default=None)
                        etype = row_get(row, ["edge_type", "type(r)"], index=1, default=edge_type)
                        weight = row_get(row, ["weight", "r.weight"], index=2, default=0.5)

                        if nid:
                            try:
                                weight_val = float(weight) if weight else 0.5
                            except (ValueError, TypeError):
                                weight_val = 0.5
                            connections.append((str(nid), str(etype), weight_val))
            except Exception:
                continue

    except Exception as exc:
        logger.warning(
            "fluid.graph_access: failed to get connections for node %s (provider=%s): %s (%s)",
            node_id,
            provider_name,
            type(exc).__name__,
            exc,
        )

    return connections


def _get_provider_name(graph_engine: Any) -> str:
    """Get a readable name for the graph provider for logging."""
    if graph_engine is None:
        return "None"
    return getattr(graph_engine, "__class__", type(graph_engine)).__name__
