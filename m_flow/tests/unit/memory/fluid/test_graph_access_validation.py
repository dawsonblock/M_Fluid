"""
Test graph_access identifier validation.
Tests that unsafe graph identifiers are rejected.
"""

from m_flow.memory.fluid.graph_access import validate_graph_identifier


def test_validate_graph_identifier_safe():
    """Test safe identifiers pass validation."""
    assert validate_graph_identifier("node_123") is True
    assert validate_graph_identifier("abc-xyz") is True
    assert validate_graph_identifier("id:123.456") is True
    assert validate_graph_identifier("MyNode_1") is True


def test_validate_graph_identifier_unsafe():
    """Test unsafe identifiers fail validation."""
    # SQL injection patterns
    assert validate_graph_identifier("'; DROP TABLE users; --") is False
    # Cypher injection patterns
    assert validate_graph_identifier("MATCH (n) DELETE n") is False
    # Empty string
    assert validate_graph_identifier("") is False
    # Special chars not in allowlist
    assert validate_graph_identifier("node@123") is False
    assert validate_graph_identifier("node$123") is False
    assert validate_graph_identifier("node`123") is False
    assert validate_graph_identifier('node"123') is False
    assert validate_graph_identifier("node'123") is False
    assert validate_graph_identifier("node(123)") is False
