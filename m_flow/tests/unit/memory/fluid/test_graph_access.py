"""
Test graph_access helper functions.
Tests row_get, get_node_text, get_neighbour_ids, get_connected_nodes
with dict-row and tuple-row graph providers.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock


def test_row_get_dict():
    """Test row_get with dict-style rows."""
    from m_flow.memory.fluid.graph_access import row_get

    row = {"neighbor_id": "node123", "edge_type": "has_facet", "weight": 0.8}
    assert row_get(row, ["neighbor_id"]) == "node123"
    assert row_get(row, ["edge_type"]) == "has_facet"
    assert row_get(row, ["missing"], default="fallback") == "fallback"


def test_row_get_tuple():
    """Test row_get with tuple-style rows."""
    from m_flow.memory.fluid.graph_access import row_get

    row = ("node123", "has_facet", 0.8)
    assert row_get(row, ["neighbor_id"], index=0) == "node123"
    assert row_get(row, ["edge_type"], index=1) == "has_facet"
    assert row_get(row, ["weight"], index=2) == 0.8
    assert row_get(row, ["missing"], index=3, default="fallback") == "fallback"


def test_row_get_object():
    """Test row_get with object-style rows."""
    from m_flow.memory.fluid.graph_access import row_get

    class MockRow:
        def __init__(self):
            self.neighbor_id = "node123"
            self.edge_type = "has_facet"
            self.weight = 0.8

    row = MockRow()
    assert row_get(row, ["neighbor_id"]) == "node123"
    assert row_get(row, ["edge_type"]) == "has_facet"


def test_row_get_none():
    """Test row_get with None row."""
    from m_flow.memory.fluid.graph_access import row_get

    assert row_get(None, ["neighbor_id"], default="fallback") == "fallback"


@pytest.mark.asyncio
async def test_get_node_text_parameterized():
    """Test get_node_text uses parameterized queries."""
    from m_flow.memory.fluid.graph_access import get_node_text

    mock_engine = AsyncMock()
    mock_engine.query = AsyncMock(return_value=[{"text": "sample text"}])

    result = await get_node_text(mock_engine, "node123")
    assert result == "sample text"

    # Verify parameterized query was called
    mock_engine.query.assert_called()
    call_args = mock_engine.query.call_args
    # Should be called with query and params dict
    assert len(call_args[0]) == 1  # query string
    assert "$node_id" in call_args[0][0]


@pytest.mark.asyncio
async def test_get_connected_nodes_tuple_rows():
    """Test get_connected_nodes handles tuple-row providers."""
    from m_flow.memory.fluid.graph_access import get_connected_nodes

    mock_engine = AsyncMock()
    # Return tuple-style rows
    mock_engine.query = AsyncMock(return_value=[
        ("node456", "has_facet", 0.7),
        ("node789", "involves_entity", 0.5),
    ])

    result = await get_connected_nodes(mock_engine, "node123")
    assert len(result) == 2
    assert result[0] == ("node456", "has_facet", 0.7)
    assert result[1] == ("node789", "involves_entity", 0.5)


@pytest.mark.asyncio
async def test_get_connected_nodes_dict_rows():
    """Test get_connected_nodes handles dict-row providers."""
    from m_flow.memory.fluid.graph_access import get_connected_nodes

    mock_engine = AsyncMock()
    # Return dict-style rows
    mock_engine.query = AsyncMock(return_value=[
        {"neighbor_id": "node456", "edge_type": "has_facet", "weight": 0.7},
        {"neighbor_id": "node789", "edge_type": "involves_entity", "weight": 0.5},
    ])

    result = await get_connected_nodes(mock_engine, "node123")
    assert len(result) == 2
    assert result[0] == ("node456", "has_facet", 0.7)
    assert result[1] == ("node789", "involves_entity", 0.5)
