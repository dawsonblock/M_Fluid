"""
Test contradiction detector safety.
Tests that structured gate fails closed on exception.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.mark.asyncio
async def test_structured_gate_fails_closed():
    """Test _check_structured_gate returns False on exception."""
    from m_flow.memory.fluid.contradiction_detector import ContradictionDetector

    mock_graph = AsyncMock()
    mock_graph.query = MagicMock(side_effect=Exception("Graph error"))

    detector = ContradictionDetector(
        graph=mock_graph,
        llm_client=None,
        config=MagicMock(
            enable_llm_contradiction=True,
            structured_contradiction_required=True,
            min_llm_contradiction_confidence=0.70,
        ),
    )

    # Should return False (fail closed) when exception occurs
    result = await detector._check_structured_gate("node_a", "node_b")
    assert result is False


@pytest.mark.asyncio
async def test_unrelated_claims_no_contradiction():
    """Test unrelated claims do not trigger contradiction."""
    from m_flow.memory.fluid.contradiction_detector import ContradictionDetector

    mock_graph = AsyncMock()
    # Return no shared connections
    from m_flow.memory.fluid.graph_access import get_connected_nodes
    from unittest.mock import patch

    async def mock_get_connected_nodes(graph, node_id, edge_types=None):
        return []

    with patch(
        "m_flow.memory.fluid.contradiction_detector.get_connected_nodes",
        side_effect=mock_get_connected_nodes,
    ):
        detector = ContradictionDetector(
            graph=mock_graph,
            llm_client=None,
            config=MagicMock(
                enable_llm_contradiction=True,
                structured_contradiction_required=True,
                min_llm_contradiction_confidence=0.70,
            ),
        )

        result = await detector._check_structured_gate("node_a", "node_b")
        assert result is False
